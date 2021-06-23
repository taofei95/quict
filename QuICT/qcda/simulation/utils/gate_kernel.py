from numba import njit, prange
import numpy as np


HG_CONSTANT = np.complex64(1/np.sqrt(2))


MASK = np.array([(1 << idx) - 1 for idx in range(31)], dtype=np.int32)
INDEX = np.array([1 << idx for idx in range(31)], dtype=np.int32)


@njit(parallel=True, nogil=True)
def Hgate_kernel(index, vector):
    for i in prange(vector.size//2):
        _0 = (i & ((1 << index) - 1)) | (i >> index << (index + 1))
        _1 = _0 | (1 << index)

        v_0, v_1 = vector[_0]*HG_CONSTANT, vector[_1]*HG_CONSTANT
        vector[_0], vector[_1] = v_0 + v_1, v_0 - v_1


@njit(parallel=True, nogil=True)
def Hgate_kernel_chunk(index, vector):
    for i in prange(vector.size//16):
        _0 = (i & MASK[index]) | (i >> index << (index + 1))
        _1 = _0 | INDEX[index]
        _0, _1 = _0 << 3, _1 << 3

        v_0, v_1 = np.multiply(vector[_0:_0+8], HG_CONSTANT), np.multiply(vector[_1:_1+8], HG_CONSTANT)
        vector[_0:_0+8], vector[_1:_1+8] = np.add(v_0, v_1), np.subtract(v_0, v_1)


@njit(parallel=True, nogil=True)
def CRZgate_kernel_target(c_index, t_index, matrix, vector):
    c_value = INDEX[c_index]
    t_value = INDEX[t_index]

    for i in prange(vector.size//4):
        gw = i >> c_index << (c_index + 1)
        _0 = c_value | (gw & MASK[t_index]) | (gw >> t_index << (t_index + 1)) | (i & MASK[c_index])
        _1 = _0 | t_value

        vector[_0], vector[_1] = vector[_0]*matrix[2,2], vector[_1]*matrix[3,3]


@njit(parallel=True, nogil=True)
def CRZgate_kernel_target_chunk(c_index, t_index, matrix, vector):
    c_index, t_index = c_index - 3, t_index - 3
    c_value = INDEX[c_index]
    t_value = INDEX[t_index]

    for i in prange(vector.size//32):
        gw = i >> c_index << (c_index + 1)
        _0 = c_value | (gw & MASK[t_index]) | (gw >> t_index << (t_index + 1)) | (i & MASK[c_index])
        _1 = _0 | t_value
        _0, _1 = _0 << 3, _1 << 3
        vector[_0:_0+8], vector[_1:_1+8] = np.multiply(vector[_0], matrix[2,2]), np.multiply(vector[_1], matrix[3,3])


@njit(parallel=True, nogil=True)
def CRZgate_kernel_control(c_index, t_index, matrix, vector):
    c_value = INDEX[c_index]
    t_value = INDEX[t_index]

    for i in prange(vector.size//4):
        gw = i >> t_index << (t_index + 1)
        _0 = c_value | (gw & MASK[c_index]) | (gw >> c_index << (c_index + 1)) | (i & MASK[t_index])
        _1 = _0 | t_value

        vector[_0], vector[_1] = vector[_0]*matrix[2,2], vector[_1]*matrix[3,3]
