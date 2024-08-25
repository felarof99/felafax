from ml_collections import ConfigDict
import cloudpickle as pickle


def create_config_dict(*args, **kwargs):
    """Creates and returns a ml_collections ConfigDict object.

    Example:
        # Using args
        config = create_config_dict({'learning_rate': 0.001, 'batch_size': 32})

        # Using kwargs
        config = create_config_dict(learning_rate=0.001, batch_size=32)

        # Using both args and kwargs
        base_config = {'model_type': 'transformer'}
        config = create_config_dict(base_config, num_layers=6, hidden_size=768)
    """
    return ConfigDict(dict(*args, **kwargs))


def open_file(path, mode="rb", cache_type="readahead"):
    if path.startswith("gs://"):
        raise NotImplementedError("GCS is not implemented yet.")
        # import gcsfs
        # logging.getLogger("fsspec").setLevel(logging.WARNING)
        # return gcsfs.GCSFileSystem().open(path, mode, cache_type=cache_type)
    else:
        return open(path, mode)


def makedirs(path, exist_ok=True):
    if path.startswith("gs://"):
        raise NotImplementedError("GCS is not implemented yet.")
        # import gcsfs
        # return gcsfs.GCSFileSystem().makedirs(path, exist_ok=exist_ok)
    else:
        return os.makedirs(path, exist_ok=exist_ok)


def save_pickle(obj, path):
    with open_file(path, "wb") as fout:
        pickle.dump(obj, fout)


def load_pickle(path):
    with open_file(path, "rb") as fin:
        data = pickle.load(fin)
    return data
