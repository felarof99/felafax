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

    def save_checkpoint_simple(self, train_state, filename):
        path = os.path.join(self.checkpoint_dir, filename)
        with utils.open_file(path, "wb") as fout:
            fout.write(
                flax.serialization.msgpack_serialize(
                    train_state.params["params"], in_place=True))

    @staticmethod
    def load_flax_checkpoint(path, target=None, shard_fns=None):
        """Load a standard flax checkpoint that's not saved with the
        msgpack streaming format.

        Args:
            path (str): Path to the checkpoint file.
            target (Any, optional): Template object to restore the state into.
            shard_fns (dict, optional): Functions to apply to each shard of the loaded state.

        Returns:
            The loaded and potentially reshaped state dictionary.
        """
        # Read the encoded checkpoint data
        with utils.open_file(path) as fin:
            encoded_bytes = fin.read()

        # Deserialize the checkpoint data using msgpack
        state_dict = flax.serialization.msgpack_restore(encoded_bytes)

        # If shard functions are provided, apply them to reshape the loaded state
        if shard_fns is not None:
            shard_fns = to_state_dict(shard_fns)
            state_dict = tree_apply(shard_fns, state_dict)

        # If no target is provided, return the state dictionary as is
        if target is None:
            return state_dict
        # Otherwise, restore the state into the provided target structure
        return from_state_dict(target, state_dict)

    @classmethod
    def load_trainstate_checkpoint(
        cls,
        load_from,
        trainstate_target=None,
        trainstate_shard_fns=None,
        disallow_trainstate=False,
    ):
        """
        Load a checkpoint for the training state.

        Args:
            load_from (str): String specifying the checkpoint type and path.
            trainstate_target (Any, optional): Template of the expected training state structure.
            trainstate_shard_fns (dict, optional): Functions to reshape the loaded state.
            disallow_trainstate (bool): If True, prevents loading full training state.

        Returns:
            tuple: (train_state, restored_params)
        """
        # Extract the parameters target from the trainstate_target if provided
        if trainstate_target is not None:
            params_target = trainstate_target.params["params"]
        else:
            params_target = None

        # Extract the parameter shard functions if provided
        if trainstate_shard_fns is not None:
            params_shard_fns = trainstate_shard_fns.params["params"]
        else:
            params_shard_fns = None

        # Split the load_from string into type and path
        load_type, load_path = load_from.split("::", 1)

        # Check if loading full trainstate is allowed
        if disallow_trainstate:
            assert load_type != "trainstate", "Loading full trainstate is not allowed!"

        train_state = None
        restored_params = None

        # Handle different checkpoint types
        if load_type == "flax_serialized":
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
