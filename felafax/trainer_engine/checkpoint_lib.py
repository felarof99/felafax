"""Checkpoints JAX models efficiently, will replace with Orbax soon!"""
import os

import flax
import jax
import jax.numpy as jnp

import msgpack
from flax.serialization import (from_bytes, from_state_dict, to_bytes,
                                to_state_dict)
from flax.traverse_util import empty_node, flatten_dict, unflatten_dict
from ml_collections import ConfigDict

from felafax.trainer_engine import utils


def get_float_dtype_by_name(dtype):
    return {
        'bf16': jnp.bfloat16,
        'bfloat16': jnp.bfloat16,
        'fp16': jnp.float16,
        'float16': jnp.float16,
        'fp32': jnp.float32,
        'float32': jnp.float32,
        'fp64': jnp.float64,
        'float64': jnp.float64,
    }[dtype]


def tree_apply(fns, tree):
    """ Apply a pytree of functions to the pytree. """
    return jax.tree_util.tree_map(lambda fn, x: fn(x), fns, tree)


def float_tensor_to_dtype(tensor, dtype):
    if dtype is None or dtype == '':
        return tensor
    if isinstance(dtype, str):
        dtype = get_float_dtype_by_name(dtype)
    float_dtypes = (jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64)
    if getattr(tensor, 'dtype', None) in float_dtypes:
        tensor = tensor.astype(dtype)
    return tensor


class Checkpointer(object):
    """Checkpoints JAX models efficiently."""

    @staticmethod
    def get_default_config(updates=None):
        config = utils.create_config_dict(float_dtype="bf16",
                                          save_optimizer_state=False)

        if updates is not None:
            config = utils.update_config_dict(config, updates)
        return config

    def __init__(self, config, checkpoint_dir, enable_checkpointer=True):
        self.config = self.get_default_config(config)
        self.checkpoint_dir = checkpoint_dir
        self.enable_checkpointer = enable_checkpointer

    def save_checkpoint_simple(self, params, filename):
        path = os.path.join(self.checkpoint_dir, filename)
        with utils.open_file(path, "wb") as fout:
            fout.write(
                flax.serialization.msgpack_serialize(params, in_place=True))

    def save_checkpoint(self, train_state, filename, gather_fns=None):
        if self.enable_checkpointer:
            path = os.path.join(self.checkpoint_dir, filename)
        else:
            path = "/dev/null"
        self.save_train_state_to_file(train_state, path, gather_fns,
                                      self.config.float_dtype)

    @staticmethod
    def save_train_state_to_file(train_state,
                                 path,
                                 gather_fns=None,
                                 float_dtype=None):
        train_state = to_state_dict(train_state)
        packer = msgpack.Packer()
        flattend_train_state = flatten_dict(train_state)
        if gather_fns is not None:
            gather_fns = flatten_dict(to_state_dict(gather_fns))

        with open(path, "wb") as fout:
            for key, value in flattend_train_state.items():
                if gather_fns is not None:
                    value = gather_fns[key](value)
                value = float_tensor_to_dtype(value, float_dtype)
                fout.write(packer.pack((key, to_bytes(value))))

    def save_all(self, train_state, gather_fns):
        step = int(jax.device_get(train_state.step))
        if self.config.save_optimizer_state:
            checkpoint_state = train_state
            checkpoint_name = "streaming_train_state"
            checkpoint_gather_fns = gather_fns
        else:
            checkpoint_state = train_state.params["params"]
            checkpoint_name = "streaming_params"
            checkpoint_gather_fns = gather_fns.params["params"]

        # Save a normal checkpoint that can be overwritten
        self.save_checkpoint(checkpoint_state, f"{checkpoint_name}",
                             checkpoint_gather_fns)

    @staticmethod
    def load_flax_checkpoint(path, target=None, shard_fns=None):
        """Load a standard flax checkpoint that's not saved with the
        msgpack streaming format.
        """
        with utils.open_file(path) as fin:
            encoded_bytes = fin.read()

        state_dict = flax.serialization.msgpack_restore(encoded_bytes)
        if shard_fns is not None:
            shard_fns = to_state_dict(shard_fns)
            state_dict = tree_apply(shard_fns, state_dict)

        if target is None:
            return state_dict
        return from_state_dict(target, state_dict)

    @classmethod
    def load_trainstate_checkpoint(
        cls,
        load_from,
        trainstate_target=None,
        trainstate_shard_fns=None,
        disallow_trainstate=False,
    ):
        if trainstate_target is not None:
            params_target = trainstate_target.params["params"]
        else:
            params_target = None

        if trainstate_shard_fns is not None:
            params_shard_fns = trainstate_shard_fns.params["params"]
        else:
            params_shard_fns = None

        load_type, load_path = load_from.split("::", 1)
        if disallow_trainstate:
            assert load_type != "trainstate", "Loading full trainstate is not allowed!"
        train_state = None
        restored_params = None
        if load_type == "flax_params":
            # Load the params in the standard flax format (non-streaming)
            # This requires the entire params to fit in memory
            restored_params = cls.load_flax_checkpoint(
                path=load_path,
                target=params_target,
                shard_fns=params_shard_fns)
            restored_params = {"params": restored_params}
        else:
            raise ValueError(f"Invalid load_from type: {load_type}")

        return train_state, restored_params
