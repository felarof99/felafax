# Standard library imports
import functools
import os

# Third-party imports
import chex
import flax
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax.training.train_state import TrainState
from jax.experimental.pjit import pjit
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as PS

from felafax.trainer_engine import checkpoint_lib, jax_utils, utils
from felafax.llama_config import LlamaFactory


class Trainer:

    def __init__(
        self,
        model,
        model_ckpt_path,
        model_factory: LlamaFactory,
        optimizer,
        training_config,
        model_params=None,
    ):
        self.model = model
        self.model_ckpt_path = model_ckpt_path
        self.model_factory = model_factory

        self.optimizer = optimizer
        self.training_config = training_config
        self.mesh = jax_utils.MESH

        self.checkpointer = checkpoint_lib.Checkpointer(
            checkpoint_lib.Checkpointer.get_default_config(),
            checkpoint_dir=os.path.dirname(model_ckpt_path),
            enable_checkpointer=jax.process_index() == 0,
        )

    def create_train_state_from_params(self, model_params):
        optimizer = self.optimizer
        train_state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=model_params,
            tx=optimizer,
        )
        return train_state

    def load_checkpoint(self):
        # You have to load checkpoint and apply sharding functions.
        self.checkpointer.load_checkpoint(self.model_ckpt_path)
