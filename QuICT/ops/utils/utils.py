import numpy as np
from numba import njit, prange

from functools import wraps
from collections.abc import Callable

# GPU Initialized
try:
    import pycuda.autoinit
    import pycuda.driver as drv

    GPU_AVAILABLE = bool(drv.Device.count())
except Exception as _:
    GPU_AVAILABLE = False


def gpu_decorator(threshold: int, cpu_func: Callable, gpu_func: Callable):
    """
    Using the gpu function when condition is satisfied.

    args:
        threshold(int|tuple): the qubit number for determining using cpu or gpu.
        gpu_func(function): the gpu function.
    """

    def decorate(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            # grap CPU parameters (without gpu-in and gpu-out)
            cpu_parameters = cpu_func.__code__.co_argcount
            cpu_args, cpu_kwargs = args[:cpu_parameters], kwargs.copy()

            if "gpu_out" in cpu_kwargs.keys():
                del cpu_kwargs["gpu_out"]

            if not GPU_AVAILABLE or threshold == -1:
                return cpu_func(*cpu_args, **cpu_kwargs)

            based_size = 1
            for var in args + tuple(kwargs.values()):
                if type(var) is np.ndarray:
                    based_size *= var.shape[0]

                    if based_size >= 1 << threshold:
                        return gpu_func(*args, **kwargs)

            return cpu_func(*cpu_args, **cpu_kwargs)

        return wrapper

    return decorate


@njit(nogil=True)
def mapping_augment(mapping: np.ndarray) -> np.ndarray:
    n = len(mapping)
    p2n = 1 << n
    res = np.zeros(shape=p2n, dtype=np.int64)
    for i in range(p2n):
        for k in range(n):
            res[i] |= ((i >> (n - 1 - mapping[k])) & 1) << (n - 1 - k)
    return res


def vector_reindex(n: int, mapping: np.ndarray) -> np.ndarray:
    n_array = np.arange(n, dtype=np.int32)
    remaining_bits = np.setdiff1d(n_array, mapping)
    mapping_value, remaining_value = np.sum(2 ** mapping), np.sum(2 ** remaining_bits)

    idx_array = np.arange(1 << n, dtype=np.int32)
    idx_array = np.bitwise_and(idx_array, remaining_value)
    idx_unique = np.unique(idx_array)
    idx_out = np.empty((1<<mapping.shape[0], 1 << n - mapping.shape[0]), dtype=np.int32)

    for i in prange(idx_unique.shape[0]):
        idx_out[:, i] = np.where(idx_array == idx_unique[i])[0]

    return idx_out
