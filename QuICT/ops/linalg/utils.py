import numpy as np
from numba import njit

from functools import wraps
from collections.abc import Callable

# GPU Initialized
try:
    import pycuda.autoinit
    import pycuda.driver as drv

    GPU_AVAILABLE = bool(drv.Device.count())
except Exception as _:
    GPU_AVAILABLE = False


def gpu_decorator(threshold: int, gpu_func: Callable):
    """
    Using the gpu function when condition is satisfied.

    args:
        threshold(int|tuple): the qubit number for determining using cpu or gpu.
        gpu_func(function): the gpu function.
    """

    def decorate(cpu_func):

        @wraps(cpu_func)
        def wrapper(*args, **kwargs):
            # grap CPU parameters (without gpu-in and gpu-out)
            cpu_parameters = cpu_func.__code__.co_argcount
            cpu_args, cpu_kwargs = args[:cpu_parameters], kwargs.copy()

            if "gpu_in" in cpu_kwargs.keys():
                del cpu_kwargs["gpu_in"]

            if "gpu_out" in cpu_kwargs.keys():
                del cpu_kwargs["gpu_out"]

            if not GPU_AVAILABLE:
                return cpu_func(*cpu_args, **cpu_kwargs)

            using_gpu = False
            for var in args:
                if type(var) is np.ndarray:
                    using_gpu = var.size > 1 << threshold

            result = gpu_func(*args, **kwargs) if using_gpu else cpu_func(*cpu_args, **cpu_kwargs)
            return result

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
