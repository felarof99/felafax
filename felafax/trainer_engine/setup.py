"""Sets up the environment for the training."""

import os


def setup_environment():
    os.environ["HF_HUB_CACHE"] = "/mnt/persistent-disk/hf/"
    os.environ["HF_HOME"] = "/mnt/persistent-disk/hf/"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Note: The following shell commands won't work directly in Python.
    # We'll use os.system to execute them.
    os.system('export HF_HUB_CACHE="/mnt/persistent-disk/hf/"')
    os.system('export HF_HOME="/mnt/persistent-disk/hf/"')
    os.system("export TOKENIZERS_PARALLELISM=false")


def main():
    setup_environment()


if __name__ == "__main__":
    main()
