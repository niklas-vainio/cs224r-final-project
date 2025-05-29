import numpy as np
import brs_algo.utils as U


def seq_chunk_collate_fn(sample_list):
    """
    sample_list: list of (T, ...). PyTorch's native collate_fn can stack all data.
    But here we also add a leading singleton dimension, so it won't break the compatibility with episode data format.
    """
    sample_list = U.any_stack(sample_list, dim=0)  # (B, T, ...)
    sample_list = nested_np_expand_dims(sample_list, axis=0)  # (1, B, T, ...)
    # convert to tensor
    return any_to_torch_tensor(sample_list)


@U.make_recursive_func
def nested_np_expand_dims(x, axis):
    if U.is_numpy(x):
        return np.expand_dims(x, axis=axis)
    else:
        raise ValueError(f"Input ({type(x)}) must be a numpy array.")


def any_to_torch_tensor(x):
    if isinstance(x, dict):
        return {k: any_to_torch_tensor(v) for k, v in x.items()}
    return U.any_to_torch_tensor(x)
