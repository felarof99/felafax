import numpy as np
import jax
import jax.numpy as jnp

from jax.sharding import PartitionSpec as PS
from jax.sharding import NamedSharding, Mesh
from jax.experimental import mesh_utils

import re

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


def make_shard_and_gather_fns(partition_specs, dtype_specs=None):
    """Creates pytree of sharding and gathering functions from pytree of partition specs."""

    float_dtypes = (jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64)

    def make_to_dtype_fn(dtype_spec):

        def to_dtype(tensor):
            if dtype_specs in float_dtypes and getattr(tensor, 'dtype',
                                                       None) in float_dtypes:
                # Convert all float tensors to the same dtype
                return tensor.astype(dtype_specs)
            elif hasattr(dtype_spec, 'dtype') and hasattr(tensor, 'dtype'):
                return tensor.astype(dtype_spec.dtype)
            return tensor

        return to_dtype

    def make_shard_fn(partition_spec, dtype_spec=None):
        jax_shard_function = jax.jit(make_to_dtype_fn(dtype_spec),
                                     in_shardings=None,
                                     out_shardings=NamedSharding(
                                         MESH, partition_spec))

        def shard_fn(tensor):
            return jax_shard_function(tensor).block_until_ready()

        return shard_fn

    def make_gather_fn(partition_spec, dtype_spec=None):
        jax_gather_fn = jax.jit(make_to_dtype_fn(dtype_spec),
                                in_shardings=NamedSharding(
                                    MESH, partition_spec),
                                out_shardings=None)

        def gather_fn(tensor):
            return jax.device_get(jax_gather_fn(tensor))

        return gather_fn

    if dtype_specs is None or dtype_specs in float_dtypes:
        shard_fns = jax.tree_util.tree_map(make_shard_fn, partition_specs)
        gather_fns = jax.tree_util.tree_map(make_gather_fn, partition_specs)
    else:
        shard_fns = jax.tree_util.tree_map(make_shard_fn, partition_specs,
                                           dtype_specs)
        gather_fns = jax.tree_util.tree_map(make_gather_fn, partition_specs,
                                            dtype_specs)
    return shard_fns, gather_fns


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


def tree_path_to_string(path, sep=None):
    """Converts a JAX tree path to a string representation.
    
    Example: tree_path_to_string([DictKey('layer1'), SequenceKey(0)], sep='/') -> 'layer1/0'
    """
    keys = []
    for key in path:
        if isinstance(key, jax.tree_util.SequenceKey):
            keys.append(str(
                key.idx))  # Use index for sequences (lists, tuples)
        elif isinstance(key, jax.tree_util.DictKey):
            keys.append(str(key.key))  # Use actual key for dictionaries
        elif isinstance(key, jax.tree_util.GetAttrKey):
            keys.append(str(key.name))  # Use attribute name for objects
        elif isinstance(key, jax.tree_util.FlattenedIndexKey):
            keys.append(str(key.key))  # Use index for flattened arrays
        else:
            keys.append(str(key))  # Fallback: convert key to string directly

    if sep is None:
        return tuple(keys)  # Return as tuple if no separator
    return sep.join(keys)  # Join with separator if provided


def flatten_tree(xs, is_leaf=None, sep=None):
    """Flattens a JAX tree into a dictionary with path strings as keys."""
    flattened, _ = jax.tree_util.tree_flatten_with_path(xs, is_leaf=is_leaf)
    output = {}
    for key, val in flattened:
        output[tree_path_to_string(key, sep=sep)] = val
    return output


def named_tree_map(f, tree, is_leaf=None, sep='/'):
    """
    Maps a function over a JAX tree, providing both path and value to the function.
    
    Args:
        f: Function to apply to each node. It should accept (path, value) as arguments.
        tree: The tree structure to map over.
        is_leaf: Optional function to determine what constitutes a leaf in the tree.
        sep: Separator used in the string representation of the path.
    
    Returns:
        A new tree with f applied to each node.
    """

    # Helper function to process each node
    def process_node(path, value):
        # Convert the path to a string
        path_str = tree_path_to_string(path, sep=sep)
        return f(path_str, value)

    # Apply our helper function to the tree
    return jax.tree_util.tree_map_with_path(process_node,
                                            tree,
                                            is_leaf=is_leaf)


def match_partition_rules(rules, params):
    """Applies partitioning rules to a parameter tree."""

    def get_partition_spec(parm_path, param_value):
        # Don't partition scalar values
        if len(param_value.shape) == 0 or np.prod(param_value.shape) == 1:
            return PS()

        for rule, ps in rules:
            if re.search(rule, parm_path) is not None:
                return ps

        raise ValueError(f'Partition rule not found for param: {parm_path}')

    # Apply get_partition_spec to each leaf in the parameter tree
    return named_tree_map(get_partition_spec, params, sep='/')
