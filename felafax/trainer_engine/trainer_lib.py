import functools
import os
import gc
import shutil
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import flax
import torch
from flax.training import train_state
from jax.sharding import Mesh, NamedSharding, PartitionSpec as PS

from . import checkpoint_lib, utils, jax_utils
from felafax.llama_train import cross_entropy_loss_and_accuracy

from transformers import LlamaConfig, LlamaForCausalLM


class FelafaxTrainer(ABC):

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def train_step(self, state, batch, rng):
        pass

    @abstractmethod
    def eval_step(self, state, batch):
        pass

    @abstractmethod
    def save_checkpoint(self, state, path):
        pass

    @abstractmethod
    def load_checkpoint(self, path):
        pass

    @abstractmethod
    def compute_loss(self, logits, labels, mask):
        pass


class CausalLMTrainer(FelafaxTrainer):

    def __init__(
        self,
        model,
        model_ckpt_path,
        model_configurator,
        optimizer,
        training_config,
        mesh,
        model_params=None,
    ):
        self.model = model
        self.model_ckpt_path = model_ckpt_path
        self.model_configurator = model_configurator
        self.optimizer = optimizer
        self.training_config = training_config
        self.mesh = mesh
        self.model_params = model_params

        self.setup()

    def setup(self):
        self.checkpointer = checkpoint_lib.Checkpointer(
            checkpoint_lib.Checkpointer.get_default_config(),
            checkpoint_dir=os.path.dirname(self.model_ckpt_path),
            enable_checkpointer=jax.process_index() == 0,
        )

        self.state_shapes = self.get_state_shapes()
        self.state_shapes_partitioned = jax_utils.match_partition_rules(
            self.model_configurator.get_partition_rules(), self.state_shapes)

        self.shard_fns, self.gather_fns = jax_utils.make_shard_and_gather_fns(
            self.state_shapes_partitioned, self.state_shapes)

        jax_utils.init_rng(99)
        jax_utils.next_rng()

        with self.mesh:
            print("Loading causal language model...")
            if self.model_params is None:
                _, self.model_params = self.checkpointer.load_trainstate_checkpoint(
                    "flax_params::" + self.model_ckpt_path, self.state_shapes,
                    self.shard_fns)

            if self.model_params is not None:
                self.train_state = self.create_train_state_from_params(
                    self.model_params)
            else:
                raise ValueError("Failed to load checkpoint")

    def get_state_shapes(self):
        return jax.eval_shape(
            functools.partial(
                self.initialize_state,
                rng=jax.random.PRNGKey(0),
                model=self.model,
                model_config=self.model_configurator,
                seq_length=self.training_config.seq_length,
                optimizer=self.optimizer,
            ))

    @staticmethod
    def initialize_state(rng, model, model_config, seq_length, optimizer):
        rng_generator = jax_utils.NextRNG(rng)

        # TODO(ntnsonti): The batch is probably hardcoded to 4 because of the 4 TPU cores, but it can be 1 as well.
        params = model.init(
            input_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((4, seq_length), dtype=jnp.int32),
            rngs=rng_generator(model_config.get_rng_keys()),
        )
        return train_state.TrainState.create(params=params,
                                             tx=optimizer,
                                             apply_fn=model.apply)

    def create_train_state_from_params(self, params):
        return train_state.TrainState.create(params=params,
                                             apply_fn=self.model.apply,
                                             tx=self.optimizer)

    @property
    def jitted_train_step(self):
        return jax.jit(
            self.train_step,
            in_shardings=(
                self.state_shapes_partitioned,  # state
                NamedSharding(self.mesh, PS()),  # batch
                NamedSharding(self.mesh, PS())  # rng
            ),
            out_shardings=(
                self.state_shapes_partitioned,  # updated state
                NamedSharding(self.mesh, PS()),  # new rng
                NamedSharding(self.mesh, PS())  # metrics
            ))

    def train_step(self, state, batch, rng):
        rng_generator = jax_utils.NextRNG(rng)

        def loss_and_accuracy(params):
            logits = state.apply_fn(
                params,
                batch["input_tokens"],
                deterministic=False,
                rngs=rng_generator(('params', 'dropout', 'fcm')),
            ).logits
            return self.compute_loss(logits, batch["target_tokens"],
                                     batch["loss_masks"])

        grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
        (loss, accuracy), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        metrics = dict(
            loss=loss,
            accuracy=accuracy,
        )
        return state, rng_generator(), metrics

    @property
    def jitted_eval_step(self):
        return jax.jit(
            self.eval_step,
            in_shardings=(
                self.state_shapes_partitioned,  # state
                NamedSharding(self.mesh, PS()),  # batch
            ),
            out_shardings=NamedSharding(self.mesh, PS())  # metrics
        )

    def eval_step(self, state, batch):
        logits = state.apply_fn(
            state.params,
            batch["input_tokens"],
            deterministic=True,
        ).logits

        loss, accuracy = self.compute_loss(logits, batch["target_tokens"],
                                           batch["loss_masks"])
        metrics = dict(
            loss=loss,
            accuracy=accuracy,
        )
        return metrics

    def train(self, train_dataloader, eval_dataloader, run_jitted=True):
        state = self.train_state

        for epoch in range(self.training_config.num_epochs):
            print(f"Starting epoch {epoch} of training...")

            for step, train_batch in enumerate(train_dataloader):
                train_batch = jax.device_put(train_batch,
                                             NamedSharding(self.mesh, PS()))

                sharded_rng = jax_utils.next_rng()

                if run_jitted:
                    state, sharded_rng, metrics = self.jitted_train_step(
                        state, train_batch, sharded_rng)
                else:
                    state, sharded_rng, metrics = self.train_step(
                        state, train_batch, sharded_rng)

                if step % self.training_config.print_every_n_steps == 0:
                    print(
                        f"Epoch {epoch}, Step {step}, Train Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}"
                    )

                if step % self.training_config.eval_every_n_steps == 0:
                    eval_metrics = self.evaluate(state, eval_dataloader)
                    print(
                        f"Epoch {epoch}, Step {step}, Eval Loss: {eval_metrics['loss']:.4f}, Accuracy: {eval_metrics['accuracy']:.4f}"
                    )

                if (self.training_config.max_steps
                        and step >= self.training_config.max_steps):
                    break

        self.train_state = state
        return state

    def evaluate(self, state, eval_dataloader, run_jitted=True):
        total_loss = 0
        total_accuracy = 0
        num_batches = 0

        for eval_batch in eval_dataloader:
            eval_batch = jax.device_put(eval_batch,
                                        NamedSharding(self.mesh, PS()))

            if run_jitted:
                metrics = self.jitted_eval_step(state, eval_batch)
            else:
                metrics = self.eval_step(state, eval_batch)
            total_loss += metrics['loss']
            total_accuracy += metrics['accuracy']
            num_batches += 1

        return {
            'loss': total_loss / num_batches,
            'accuracy': total_accuracy / num_batches
        }

    def save_checkpoint(self, state, path):
        print(f"Saving checkpoint to {path}...")
        self.checkpointer.save_checkpoint_simple(train_state=state,
                                                 filename=path)
        print(f"Checkpoint saved to {path}.")

    def load_checkpoint(self, path):
        _, params = self.checkpointer.load_trainstate_checkpoint(
            path, self.state_shapes, self.shard_fns)
        return self.create_train_state_from_params(params)

    def compute_loss(self, logits, labels, mask):
        return cross_entropy_loss_and_accuracy(logits, labels, mask)

    def save_hf_compatible_checkpoint(self, model_params, save_path):
        print(f"Saving HuggingFace-compatible checkpoint to {save_path}...")
        os.makedirs(save_path, exist_ok=True)
        tmp_model_path = os.path.join(save_path, "tmp")
        os.makedirs(tmp_model_path, exist_ok=True)

        # Use shard_fns to properly flatten the model_params
        gathered_params = checkpoint_lib.tree_apply(self.gather_fns,
                                                    model_params)
        flax_params = flax.traverse_util.flatten_dict(gathered_params, sep='.')

        # Convert Flax params to PyTorch
        torch_params = {}
        for key, tensor in flax_params.items():
            if isinstance(tensor, (bool, int, float)):
                # Skip non-tensor values
                continue
            if "kernel" in key and "norm" not in key and 'ln_f' not in key:
                tensor = tensor.T
            torch_params[key] = torch.tensor(
                checkpoint_lib.float_tensor_to_dtype(tensor, 'fp32'),
                dtype=torch.float16)

        # Get model config
        llama_config = self.model_configurator.get_hf_pretrained_config(
            self.model_configurator.get_model_config())
        n_layers = llama_config.num_hidden_layers
        n_heads = llama_config.num_attention_heads
        n_kv_heads = llama_config.num_key_value_heads
        dim = llama_config.hidden_size
        dims_per_head = dim // n_heads
        base = llama_config.rope_theta
        inv_freq = 1.0 / (base**(torch.arange(0, dims_per_head, 2).float() /
                                 dims_per_head))

        param_count = 0
        index_dict = {"weight_map": {}}

        def permute(w, n_heads, input_dim, output_dim):
            return w.view(n_heads, output_dim // n_heads // 2, 2,
                          input_dim).transpose(1, 2).reshape(
                              output_dim, input_dim)

        for layer_i in range(n_layers):
            filename = f"pytorch_model-{layer_i + 1}-of-{n_layers + 1}.bin"
            state_dict = {
                f"model.layers.{layer_i}.self_attn.q_proj.weight":
                permute(
                    torch_params[
                        f"transformer.h.{layer_i}.attention.wq.kernel"],
                    llama_config.num_attention_heads,
                    llama_config.hidden_size,
                    llama_config.hidden_size,
                ),
                f"model.layers.{layer_i}.self_attn.k_proj.weight":
                permute(
                    torch_params[
                        f"transformer.h.{layer_i}.attention.wk.kernel"],
                    llama_config.num_key_value_heads,
                    llama_config.hidden_size,
                    llama_config.hidden_size //
                    (llama_config.num_attention_heads //
                     llama_config.num_key_value_heads),
                ),
                f"model.layers.{layer_i}.self_attn.v_proj.weight":
                torch_params[f"transformer.h.{layer_i}.attention.wv.kernel"],
                f"model.layers.{layer_i}.self_attn.o_proj.weight":
                torch_params[f"transformer.h.{layer_i}.attention.wo.kernel"],
                f"model.layers.{layer_i}.mlp.gate_proj.weight":
                torch_params[
                    f"transformer.h.{layer_i}.feed_forward.w1.kernel"],
                f"model.layers.{layer_i}.mlp.down_proj.weight":
                torch_params[
                    f"transformer.h.{layer_i}.feed_forward.w2.kernel"],
                f"model.layers.{layer_i}.mlp.up_proj.weight":
                torch_params[
                    f"transformer.h.{layer_i}.feed_forward.w3.kernel"],
                f"model.layers.{layer_i}.input_layernorm.weight":
                torch_params[f"transformer.h.{layer_i}.attention_norm.kernel"],
                f"model.layers.{layer_i}.post_attention_layernorm.weight":
                torch_params[f"transformer.h.{layer_i}.ffn_norm.kernel"],
            }

            state_dict[
                f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = inv_freq
            for k, v in state_dict.items():
                index_dict["weight_map"][k] = filename
                param_count += v.numel()
            torch.save(state_dict, os.path.join(tmp_model_path, filename))

        filename = f"pytorch_model-{n_layers + 1}-of-{n_layers + 1}.bin"
        # Unsharded
        state_dict = {
            "model.embed_tokens.weight":
            torch_params["transformer.wte.embedding"],
            "model.norm.weight": torch_params["transformer.ln_f.kernel"],
            "lm_head.weight": torch_params["lm_head.kernel"],
        }

        for k, v in state_dict.items():
            index_dict["weight_map"][k] = filename
            param_count += v.numel()
        torch.save(state_dict, os.path.join(tmp_model_path, filename))

        # Write configs
        index_dict["metadata"] = {"total_size": param_count * 2}
        with open(os.path.join(tmp_model_path, "pytorch_model.bin.index.json"),
                  "w") as f:
            json.dump(index_dict, f)

        config = LlamaConfig(
            vocab_size=llama_config.vocab_size,
            hidden_size=llama_config.hidden_size,
            intermediate_size=llama_config.intermediate_size,
            num_hidden_layers=llama_config.num_hidden_layers,
            num_attention_heads=llama_config.num_attention_heads,
            num_key_value_heads=llama_config.num_key_value_heads,
            initializer_range=llama_config.initializer_range,
            rms_norm_eps=llama_config.rms_norm_eps,
            max_position_embeddings=llama_config.max_position_embeddings,
            rope_theta=llama_config.rope_theta,
        )
        config.save_pretrained(tmp_model_path)

        # Make space so we can load the model properly now.
        del state_dict
        del torch_params
        gc.collect()

        print("Loading the checkpoint in a Llama model.")
        model = LlamaForCausalLM.from_pretrained(tmp_model_path,
                                                 torch_dtype=torch.float16)
        # Avoid saving this as part of the config.
        del model.config._name_or_path

        print("Saving in the Transformers format.")
        model.save_pretrained(save_path)
        shutil.rmtree(tmp_model_path)

        print(f"HuggingFace-compatible checkpoint saved to {save_path}.")
