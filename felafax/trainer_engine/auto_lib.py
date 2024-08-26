from typing import Tuple

from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

MODEL_NAME_TO_DOWNLOAD_CONFIG = {
    "llama-3.1-8B-JAX": {
        "hf_model_name": "meta-llama/Meta-Llama-3.1-8B",
        "felafax_model_name": "felafax/llama-3.1-8B-JAX",
    },
}


class AutoJAXModelForCausalLM:

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        huggingface_token: str,
        **kwargs,
    ) -> Tuple[str, AutoConfig, AutoTokenizer]:
        """Downloads the model from HF and returns the downloaded model path, config, and tokenizer."""

        print(f"Downloading model {model_name}...")
        try:
            download_config = MODEL_NAME_TO_DOWNLOAD_CONFIG[model_name]
        except KeyError:
            raise ValueError(
                f"Invalid model name: {model_name}. "
                f"Available models are: {', '.join(MODEL_NAME_TO_DOWNLOAD_CONFIG.keys())}"
            )

        config = AutoConfig.from_pretrained(download_config["hf_model_name"],
                                            token=huggingface_token)

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
        return model_path, config, tokenizer
