from .trainer_engine import utils
from jax.sharding import PartitionSpec as PS
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput


class LlamaConfigurator:
    """Manages LLaMA configuration."""

    def __init__(self, model_name):
        self.base_config = {
            "base_model": "llama_test",
            "vocab_size": 32000,
            "hidden_size": 3200,
            "intermediate_size": 8640,
            "num_hidden_layers": 26,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "initializer_range": 0.02,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 2048,
            "rope_theta": 1e4,
            "embedding_dropout": 0.0,
            "attention_dropout": 0.0,
            "residue_dropout": 0.0,
        }

        self.model_configs = {
            "llama3_8b": {
                "base_model": "llama3_8b",
                "vocab_size": 128256,
                "hidden_size": 4096,
                "intermediate_size": 14336,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "max_position_embeddings": 8192,
                "rms_norm_eps": 1e-5,
                "rope_theta": 5e5,
            },
            "llama3_70b": {
                "base_model": "llama3_8b",
                "vocab_size": 128256,
                "hidden_size": 8192,
                "intermediate_size": 28672,
                "num_hidden_layers": 80,
                "num_attention_heads": 64,
                "num_key_value_heads": 8,
                "max_position_embeddings": 8192,
                "rms_norm_eps": 1e-5,
                "rope_theta": 5e5,
            },
        }

        self.model_config = utils.create_config_dict(self.base_config)
        if model_name in self.model_configs:
            self.model_config.update(self.model_configs[model_name])

    def get_model_config(self):
        return self.model_config

    def get_hf_pretrained_config(self, config):
        """Apply updates on top of standard base model config."""
        # This is where you get pretrained config from huggingface merged with your updates.
        updated_config = config.copy()
        return PretrainedConfig.from_dict(updated_config)

    def rng_keys(self):
        return ("params", "dropout", "fcm")

    def get_partition_rules(self):
        """Parition rules for GPTJ. Note that these rules are orderd, so that
        the beginning rules match first. It is important to use
        PartitionSpec() instead of None here because JAX does not treat
        None as a pytree leaf.
        """
        return (
            # embeddings
            ("transformer/wte/embedding", PS("mp", "fsdp")),
            # atention
            ("attention/(wq|wk|wv)/kernel", PS("fsdp", "mp")),
            ("attention/wo/kernel", PS("mp", "fsdp")),
            # mlp
            ("feed_forward/w1/kernel", PS("fsdp", "mp")),
            ("feed_forward/w2/kernel", PS("mp", "fsdp")),
            ("feed_forward/w3/kernel", PS("fsdp", "mp")),
            # layer norms
            ("attention_norm/kernel", PS(None)),
            ("ffn_norm/kernel", PS(None)),
            # output head
            ("transformer/ln_f/kernel", PS(None)),
            ("lm_head/kernel", PS("fsdp", "mp")),
            (".*", PS(None)),
        )
