from typing import Tuple, Union, Dict

from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer
from ..llama_config import create_llama_factory, Llama3_1_8B, Llama3_1_70B, LlamaTest
from .. import llama_model
import jax.numpy as jnp

LlamaConfigType = Union[Dict, Llama3_1_8B, Llama3_1_70B, LlamaTest]

MODEL_NAME_TO_DOWNLOAD_CONFIG = {
    "llama-3.1-8B-JAX": {
        "hf_model_name": "meta-llama/Meta-Llama-3.1-8B",
        "felafax_model_name": "felafax/llama-3.1-8B-JAX",
        "llama_config_id": "llama3.1_8b",
    },
}


class AutoJAXModelForCausalLM:

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        huggingface_token: str,
        **kwargs,
    ) -> Tuple[str, llama_model.CausalLlamaModule, LlamaConfigType,
               AutoTokenizer]:
        """Downloads the model from HF and returns the downloaded model path, model, llama config, and tokenizer."""

        print(f"Downloading model {model_name}...")
        try:
            download_config = MODEL_NAME_TO_DOWNLOAD_CONFIG[model_name]
        except KeyError:
            raise ValueError(
                f"Invalid model name: {model_name}. "
                f"Available models are: {', '.join(MODEL_NAME_TO_DOWNLOAD_CONFIG.keys())}"
            )

        hf_config = AutoConfig.from_pretrained(
            download_config["hf_model_name"], token=huggingface_token)

        tokenizer = AutoTokenizer.from_pretrained(
            download_config["hf_model_name"],
            token=huggingface_token,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_path = snapshot_download(
            repo_id=download_config["felafax_model_name"],
            token=huggingface_token,
        )

        print(f"{model_name} was downloaded to {model_path}.")

        # Create LlamaFactory and model
        llama_factory = create_llama_factory(
            download_config["llama_config_id"])
        llama_config = llama_factory.get_model_config()
        hf_pretrained_llama_config = llama_factory.get_hf_pretrained_config(
            llama_config)

        model = llama_model.CausalLlamaModule(
            hf_pretrained_llama_config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )

        return model_path, model, llama_config, tokenizer
