import jax
import jax.numpy as jnp

from jax.sharding import PartitionSpec as PS
from jax.sharding import NamedSharding, Mesh
from jax.experimental import mesh_utils

###################################################
# Util functions for JAX RNG handling
###################################################
rng_generator = None


def init_rng(seed):
    global rng_generator
    rng_generator = NextRNG.from_seed(seed)


def next_rng(*args, **kwargs):
    global rng_generator
    return rng_generator(*args, **kwargs)


class NextRNG(object):
    """Stateful RNG generator, generate and delete within pure function."""

    @classmethod
    def from_seed(cls, seed):
        # Create new instance from a seed value
        return cls(jax.random.PRNGKey(seed))

    def __init__(self, rng):
        # Initialize with a JAX PRNG key
        self.rng = rng

    def __call__(self, keys=None):
        """Generates new RNG keys when the instance is called as a function."""
        if keys is None:
            # If no keys are provided, split the current RNG into two
            # Update the instance's RNG and return the new split RNG
            self.rng, split_rng = jax.random.split(self.rng)
            return split_rng

        elif isinstance(keys, int):
            # If an integer is provided, split the RNG into that many new keys plus one
            split_rngs = jax.random.split(self.rng, num=keys + 1)

            # Update the instance's RNG with the first split
            self.rng = split_rngs[0]

            # Return the remaining splits as a tuple
            return tuple(split_rngs[1:])
        else:
            # If a sequence of keys is provided, split the RNG into that many new keys plus one
            split_rngs = jax.random.split(self.rng, num=len(keys) + 1)

            # Update the instance's RNG with the first split
            self.rng = split_rngs[0]

            # Return a dictionary mapping the provided keys to the new RNG splits
            return {key: val for key, val in zip(keys, split_rngs[1:])}


###################################################
# Utils for JAX sharding
###################################################
# TODO: avoid defining mesh globally.
DEVICES = jax.devices()
DEVICE_COUNT = len(DEVICES)
DEVICE_MESH = mesh_utils.create_device_mesh((1, DEVICE_COUNT, 1))
MESH = Mesh(devices=DEVICE_MESH, axis_names=("dp", "fsdp", "mp"))


def apply_sharding_constraint(x, partition_spec):
    return jax.lax.with_sharding_constraint(
        x, NamedSharding(MESH, partition_spec))
