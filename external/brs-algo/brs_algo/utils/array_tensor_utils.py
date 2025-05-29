"""
Functions that work on nested structures of torch.Tensor or numpy array
"""

import functools
import torch
import numpy as np
import tree
from typing import List, Any, Union, Optional, Dict
from .tree_utils import (
    copy_non_leaf,
    tree_assign_at_path,
    is_sequence,
    tree_value_at_path,
)
from .functional_utils import meta_decorator


def is_array_tensor(obj):
    return isinstance(obj, (np.ndarray, torch.Tensor))


def is_numpy(obj):
    return isinstance(obj, np.ndarray)


def is_tensor(obj):
    return torch.is_tensor(obj)


@meta_decorator
def make_recursive_func(fn, *, with_path=False):
    """
    Decorator that turns a function that works on a single array/tensor to working on
    arbitrary nested structures.
    """

    @functools.wraps(fn)
    def _wrapper(tensor_struct, *args, **kwargs):
        if with_path:
            return tree.map_structure_with_path(
                lambda paths, x: fn(paths, x, *args, **kwargs), tensor_struct
            )
        else:
            return tree.map_structure(lambda x: fn(x, *args, **kwargs), tensor_struct)

    return _wrapper


def any_stack(xs: List, *, dim: int = 0):
    """
    Works for both torch Tensor and numpy array
    """

    def _any_stack_helper(*xs):
        x = xs[0]
        if isinstance(x, np.ndarray):
            return np.stack(xs, axis=dim)
        elif torch.is_tensor(x):
            return torch.stack(xs, dim=dim)
        elif isinstance(x, float):
            # special treatment for float, defaults to float32
            return np.array(xs, dtype=np.float32)
        else:
            return np.array(xs)

    return tree.map_structure(_any_stack_helper, *xs)


def any_concat(xs: List, *, dim: int = 0):
    """
    Works for both torch Tensor and numpy array
    """

    def _any_concat_helper(*xs):
        x = xs[0]
        if isinstance(x, np.ndarray):
            return np.concatenate(xs, axis=dim)
        elif torch.is_tensor(x):
            return torch.cat(xs, dim=dim)
        elif isinstance(x, float):
            # special treatment for float, defaults to float32
            return np.array(xs, dtype=np.float32)
        else:
            return np.array(xs)

    return tree.map_structure(_any_concat_helper, *xs)


def any_chunk(x, chunks: int, *, dim: int = 0, strict: bool = True) -> List[Any]:
    """
    Works for both torch Tensor and numpy array

    Returns:
        list of chunked nested structures
    """
    assert chunks >= 1

    x_copies = [copy_non_leaf(x) for _ in range(chunks)]

    def _any_chunk_helper(path, x):
        if is_array_tensor(x):
            if isinstance(x, np.ndarray):
                chunked_values = np.split(x, chunks, axis=dim)
            else:
                chunked_values = torch.chunk(x, chunks, dim=dim)

            if path:
                for xc, chunked in zip(x_copies, chunked_values):
                    tree_assign_at_path(xc, path, chunked)
            else:  # top-level, no nested path
                for i, chunked in enumerate(chunked_values):
                    x_copies[i] = chunked
        else:
            if strict:
                raise NotImplementedError(f"Cannot chunk type {type(x)}")
            else:
                return

    tree.map_structure_with_path(_any_chunk_helper, x)
    return x_copies


def chunk_seq(arr, chunks: int, check_divide=True):
    """
    Args:
        check_divide: True to force arr must divide n
    """
    k, m = divmod(len(arr), chunks)
    if check_divide and m != 0:
        raise ValueError(f"Array len {len(arr)} does not divide chunks {chunks}")
    return (arr[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(chunks))


@make_recursive_func
def any_zeros_like(x: Union[Dict, np.ndarray, torch.Tensor, int, float, np.number]):
    """Returns a zero-filled object of the same (d)type and shape as the input.

    The difference between this and `np.zeros_like()` is that this works well
    with `np.number`, `int`, `float`, and `jax.numpy.DeviceArray` objects without
    converting them to `np.ndarray`s.

    Args:
      x: The object to replace with 0s.

    Returns:
      A zero-filed object of the same (d)type and shape as the input.
    """
    if isinstance(x, (int, float, np.number)):
        return type(x)(0)
    elif is_tensor(x):
        return torch.zeros_like(x)
    elif is_numpy(x):
        return np.zeros_like(x)
    else:
        raise ValueError(
            f"Input ({type(x)}) must be either a numpy array, a tensor, an int, or a float."
        )


@make_recursive_func
def any_ones_like(x: Union[Dict, np.ndarray, torch.Tensor, int, float, np.number]):
    """Returns a one-filled object of the same (d)type and shape as the input.
    The difference between this and `np.ones_like()` is that this works well
    with `np.number`, `int`, `float`, and `jax.numpy.DeviceArray` objects without
    converting them to `np.ndarray`s.
    Args:
      x: The object to replace with 1s.
    Returns:
      A one-filed object of the same (d)type and shape as the input.
    """
    if isinstance(x, (int, float, np.number)):
        return type(x)(1)
    elif is_tensor(x):
        return torch.ones_like(x)
    elif is_numpy(x):
        return np.ones_like(x)
    else:
        raise ValueError(
            f"Input ({type(x)}) must be either a numpy array, a tensor, an int, or a float."
        )


@make_recursive_func
def any_zero_(x: Union[Dict, np.ndarray, torch.Tensor]):
    """
    Apply in-place zero-out to a tensor, i.e. x.zero_()
    """
    if is_tensor(x):
        x.zero_()
    elif is_numpy(x):
        x.fill(0)
    else:
        raise ValueError(f"Input ({type(x)}) must be either a numpy array or a tensor")


@make_recursive_func
def any_fill_(x: Union[Dict, np.ndarray, torch.Tensor], value):
    """
    Apply in-place zero-out to a tensor, i.e. x.zero_()
    """
    if is_tensor(x):
        x.fill_(value)
    elif is_numpy(x):
        x.fill(value)
    else:
        raise ValueError(f"Input ({type(x)}) must be either a numpy array or a tensor")


def get_batch_size(x, strict: bool = False) -> int:
    """
    Args:
        x: can be any arbitrary nested structure of np array and torch tensor
        strict: True to check all batch sizes are the same
    """

    def _get_batch_size(x):
        if isinstance(x, np.ndarray):
            return x.shape[0]
        elif torch.is_tensor(x):
            return x.size(0)
        else:
            return len(x)

    xs = tree.flatten(x)

    if strict:
        batch_sizes = [_get_batch_size(x) for x in xs]
        assert all(
            b == batch_sizes[0] for b in batch_sizes
        ), f"batch sizes must all be the same in nested structure: {batch_sizes}"
        return batch_sizes[0]
    else:
        return _get_batch_size(xs[0])


@make_recursive_func
def add_batch_dim(x):
    if is_numpy(x):
        return np.expand_dims(x, axis=0)
    elif is_tensor(x):
        return x.unsqueeze(0)
    else:
        raise NotImplementedError(f"Unsupported data structure: {type(x)}")


@make_recursive_func
def remove_batch_dim(x):
    if is_numpy(x):
        return np.squeeze(x, axis=0)
    elif is_tensor(x):
        return x.squeeze(0)
    else:
        raise NotImplementedError(f"Unsupported data structure: {type(x)}")


@make_recursive_func
def any_to_primitive(x):
    if isinstance(x, (np.ndarray, np.number, torch.Tensor)):
        return x.tolist()
    else:
        return x


@make_recursive_func
def any_get_shape(x):
    if is_numpy(x):
        return tuple(x.shape)
    elif is_tensor(x):
        return tuple(x.size())
    else:
        raise NotImplementedError(f"Unsupported data structure: {type(x)}")


@make_recursive_func
def any_mean(x, dim: Optional[int] = None, keepdim: bool = False):
    if is_numpy(x):
        return np.mean(x, axis=dim, keepdims=keepdim)
    elif is_tensor(x):
        return torch.mean(x, dim=dim, keepdim=keepdim)
    else:
        raise NotImplementedError(f"Unsupported data structure: {type(x)}")


@make_recursive_func
def any_variance(
    x, dim: Optional[int] = None, keepdim: bool = False, unbiased: bool = False
):
    if is_numpy(x):
        return np.var(x, axis=dim, keepdims=keepdim, ddof=1 if unbiased else 0)
    elif is_tensor(x):
        return torch.var(x, dim=dim, keepdim=keepdim, unbiased=unbiased)
    else:
        raise NotImplementedError(f"Unsupported data structure: {type(x)}")


@make_recursive_func
def any_describe_str(x, shape_only=False):
    """
    Describe type, shape, device, data type (of np array/tensor)
    Very useful for debugging
    """
    t = type(x)
    tname = type(x).__name__
    if is_numpy(x):
        shape = list(x.shape)
        if x.size == 1:
            if shape_only:
                return f"np scalar: {x.item()} {shape}"
            else:
                return f"np scalar: {x.item()} {shape} {x.dtype}"
        else:
            if shape_only:
                return f"np: {shape}"
            else:
                return f"np: {shape} {x.dtype}"
    elif is_tensor(x):
        shape = list(x.size())
        if x.numel() == 1:
            if shape_only:
                return f"torch scalar: {x.item()} {shape}"
            else:
                return f"torch scalar: {x.item()} {shape} {x.dtype} {x.device}"
        else:
            if shape_only:
                return f"torch: {shape}"
            else:
                return f"torch: {shape} {x.dtype} {x.device}"
    elif is_sequence(x):
        return f"{tname}[{len(x)}]"
    elif isinstance(x, str):
        return x
    elif x is None:
        return f"None"
    elif np.issubdtype(t, np.number) or np.issubdtype(t, np.bool_):
        return f"{tname}: {x}"
    else:
        return f"{tname}"


def any_describe(x, msg="", *, shape_only=False):
    # from omlet.utils import yaml_dumps
    from pprint import pprint

    if isinstance(x, str) and msg != "":
        x, msg = msg, x

    if msg:
        msg += ": "
    print(msg, end="")
    pprint(any_describe_str(x, shape_only=shape_only))


@make_recursive_func
def any_slice(x, slice):
    """
    Args:
        slice: you can use np.s_[...] to return the slice object
    """
    if is_array_tensor(x):
        return x[slice]
    else:
        return x


def any_assign(x, assign_value, slice):
    """
    Recursive version of x[slice] = assign_value
    If structures of x and assign_value do not match, we will respect `assign_value`
    E.g. x = {'a': ..., 'b': ...}, assign_value = {'a': ...}, then 'b' will not change

    Use np.s_[...] to get advanced slicing
    """

    def _any_assign_helper(path, v):
        y = tree_value_at_path(x, path)
        y[slice] = v

    tree.map_structure_with_path(_any_assign_helper, assign_value)


@make_recursive_func
def any_transpose_first_two_axes(x):
    """
    util to convert between (L, B, ...) and (B, L, ...)
    """
    if is_numpy(x):
        return np.swapaxes(x, 0, 1)
    elif is_tensor(x):
        return torch.swapaxes(x, 0, 1)
    else:
        raise ValueError(f"Input ({type(x)}) must be either a numpy array or a tensor.")


@make_recursive_func
def any_flatten_first_n_dims(x, n: int):
    """
    Flatten the first n dimensions of a tensor, keep the rest shape the same
    """
    if is_numpy(x):
        remaining_shape = x.shape[n:]
        return x.reshape((-1,) + remaining_shape)
    elif is_tensor(x):
        remaining_shape = x.size()[n:]
        return x.reshape((-1,) + remaining_shape)
    else:
        raise ValueError(f"Input ({type(x)}) must be either a numpy array or a tensor.")
