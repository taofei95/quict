from numba import njit, prange, vectorize
import numpy as np


HG_CONSTANT = np.complex64(1/np.sqrt(2))


@njit(parallel=True, nogil=True)
def Hgate_kernel(index, vector):
    for i in prange(vector.size//2):
        _0 = (i & ((1 << index) - 1)) | (i >> index << (index + 1))
        _1 = _0 | (1 << index)

        v_0, v_1 = vector[_0]*HG_CONSTANT, vector[_1]*HG_CONSTANT
        vector[_0], vector[_1] = v_0 + v_1, v_0 - v_1


@vectorize(cache=True)
def Hgate_kernel_vectorize_add(_0, _1):
    return _0*HG_CONSTANT + _1*HG_CONSTANT


@vectorize(cache=True)
def Hgate_kernel_vectorize_sub(_0, _1):
    return _0*HG_CONSTANT - _1*HG_CONSTANT


@njit(parallel=True, nogil=True)
def Hgate_kernel_chunk(index, vector):
    for i in prange(vector.size//(1 << 4)):
        ci = i << 3
        _0 = (ci & ((1 << index) - 1)) | (ci >> index << (index + 1))
        _1 = _0 | (1 << index)

        vector[_0:_0+8], vector[_1:_1+8] = Hgate_kernel_vectorize_add(vector[_0:_0+8], vector[_1:_1+8]), Hgate_kernel_vectorize_sub(vector[_0:_0+8], vector[_1:_1+8])


@njit(parallel=True, nogil=True)
def CRZgate_kernel_target(c_index, t_index, matrix, vector):
    c_value = 1 << c_index
    t_value = 1 << t_index

    for i in prange(vector.size//4):
        gw = i >> c_index << (c_index + 1)
        _0 = c_value | (gw & (t_value - c_value)) | (gw >> t_index << (t_index + 1)) | (i & (c_value - 1))
        _1 = i | t_value

        vector[_0], vector[_1] = vector[_0]*matrix[2,2], vector[_1]*matrix[3,3]


@njit(parallel=True, nogil=True)
def CRZgate_kernel_control(c_index, t_index, matrix, vector):
    c_value = 1 << c_index
    t_value = 1 << t_index

    for i in prange(vector.size//4):
        gw = i >> t_index << (t_index + 1)
        _0 = c_value | (gw & (c_value - t_value)) | (gw >> c_index << (c_index + 1)) | (i & (t_value - 1))
        _1 = i | t_value

        vector[_0], vector[_1] = vector[_0]*matrix[2,2], vector[_1]*matrix[3,3]
