import cupy as cp
import numpy as np
import random


__outward_functions = [
    "Simple_Multiply"
]


Simple_Multiply_single = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void SimpleMultiply(int value, complex<float>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        vec[label] = vec[label]*value;
    }
    ''', 'SimpleMultiply')


Simple_Multiply_double = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void SimpleMultiply(int value, complex<double>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        vec[label] = vec[label]*value;
    }
    ''', 'SimpleMultiply')


MeasureGate0_single_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void MeasureGate0Single(const int index, const float generation, complex<float>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;
        int _0 = (label & ((1 << index) - 1)) 
                + (label >> index << (index + 1));
        vec[_0] = vec[_0] * generation;
        vec[_0 + (1 << index)] = complex<float>(0, 0);
    }
    ''', 'MeasureGate0Single')


MeasureGate1_single_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void MeasureGate1Single(const int index, const float generation, complex<float>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;
        int _0 = (label & ((1 << index) - 1)) 
                + (label >> index << (index + 1));
        int _1 = _0 + (1 << index);
        vec[_0] = complex<float>(0, 0);
        vec[_1] = vec[_1] * generation;
    }
    ''', 'MeasureGate1Single')


MeasureGate0_double_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void MeasureGate0Double(const int index, const double generation, complex<double>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;
        int _0 = (label & ((1 << index) - 1)) 
                + (label >> index << (index + 1));
        vec[_0] = vec[_0] * generation;
        vec[_0 + (1 << index)] = complex<double>(0, 0);
    }
    ''', 'MeasureGate0Double')


MeasureGate1_double_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void MeasureGate1Double(const int index, const double generation, complex<double>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;
        int _0 = (label & ((1 << index) - 1)) 
                + (label >> index << (index + 1));
        int _1 = _0 + (1 << index);
        vec[_0] = complex<double>(0, 0);
        vec[_1] = vec[_1] * generation;
    }
    ''', 'MeasureGate1Double')


prop_add = cp.ElementwiseKernel(
    'T x, raw T y, int32 index', 'T z',
    'z = (i & index) ? 0 : abs(x) * abs(x)',
    'prop_add')


prob_1 = cp.ElementwiseKernel(
    'T x, raw T y', 'T z',
    'z = abs(x) * abs(x)',
    'prop_add')


MeasureGate_prop_kernel = cp.ReductionKernel(
    'T x',
    'T y',
    'x',
    'a + b',
    'y = abs(a)',
    '0',
    'MeasureGate_prop_kernel')


def Simple_Multiply(value, vec, vec_bit, sync: bool = False):
    task_number = 1 << vec_bit
    thread_per_block = min(256, task_number)
    block_num = task_number // thread_per_block

    if vec.dtype == np.complex64:
        Simple_Multiply_single(
            (block_num,),
            (thread_per_block,),
            (value, vec)
        )
    else:
        Simple_Multiply_double(
            (block_num,),
            (thread_per_block,),
            (value, vec)
        )

    if sync:
        cp.cuda.Device().synchronize()


def Device_Prob_Calculator(index, vec, device_qubits, rank):
    """
    Measure Gate Measure.
    """
    if index >= device_qubits:
        if rank & (1 << (index - device_qubits)):
            prob = cp.array([0], dtype=vec.dtype)
        else:
            prob = prob_1(vec, vec)
    else:
        prob = prop_add(vec, vec, 1 << index)
        prob = MeasureGate_prop_kernel(prob, axis = 0).real

    return prob


def Multi_MeasureGate(prob, index, vec, vec_bit, sync: bool = False):
    _0 = random.random()
    _1 = _0 > prob
    prob = prob.get()

    task_number = 1 << (vec_bit - 1)
    thread_per_block = min(256, task_number)
    block_num = task_number // thread_per_block

    if not _1:
        if vec.dtype == np.complex64:
            alpha = np.float32(1 / np.sqrt(prob))
            MeasureGate0_single_kernel(
                (block_num, ),
                (thread_per_block, ),
                (index, alpha, vec)
            )
        else:
            alpha = np.float64(1 / np.sqrt(prob))
            MeasureGate0_double_kernel(
                (block_num,),
                (thread_per_block,),
                (index, alpha, vec)
            )
    else:
        if vec.dtype == np.complex64:
            alpha = np.float32(1 / np.sqrt(1 - prob))
            MeasureGate1_single_kernel(
                (block_num,),
                (thread_per_block,),
                (index, alpha, vec)
            )
        else:
            alpha = np.float64(1 / np.sqrt(1 - prob))
            MeasureGate1_double_kernel(
                (block_num,),
                (thread_per_block,),
                (index, alpha, vec)
            )
    if sync:
        cp.cuda.Device().synchronize()

    return _1


def Multi_MeasureGate_Extra(prob, device_index, vec, vec_bit, sync: bool = False):
    _0 = random.random()
    _1 = _0 > prob
    prob = prob.get()

    if not _1:
        alpha = np.float32(1 / np.sqrt(prob)) if vec.dtype == np.complex64 else \
            np.float64(1 / np.sqrt(prob))

        if device_index:
            Set_Zeros(vec, vec_bit, sync)
        else:
            Simple_Multiply(alpha, vec, vec_bit, sync)
    else:
        alpha = np.float32(1 / np.sqrt(1 - prob)) if vec.dtype == np.complex64 else \
            np.float64(1 / np.sqrt(1 - prob))
        
        if device_index:
            Simple_Multiply(alpha, vec, vec_bit, sync)
        else:
            Set_Zeros(vec, vec_bit, sync)

    if sync:
        cp.cuda.Device().synchronize()

    return _1


SetZeros_single = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void SetZeros(const int index, const float generation, complex<float>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        vec[label] = complex<float>(0, 0);
    }
    ''', 'SetZeros')


SetZeros_double = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void SetZeros(complex<double>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        vec[label] = complex<double>(0, 0);
    }
    ''', 'SetZeros')


def Set_Zeros(vec, vec_bit, sync: bool = False):
    task_number = 1 << vec_bit
    thread_per_block = min(256, task_number)
    block_num = task_number // thread_per_block

    if vec.dtype == np.complex64:
        SetZeros_single(
            (block_num,),
            (thread_per_block,),
            (vec)
        )
    else:
        SetZeros_double(
            (block_num,),
            (thread_per_block,),
            (vec)
        )

    if sync:
        cp.cuda.Device().synchronize()


def Multi_ResetGate(prob, device_index, vec, vec_bit, sync: bool = False):
    prob = prob.get()
    alpha = 1 / np.sqrt(prob)

    if alpha < 1e-6:
        if not device_index:
            Set_Zeros(vec, vec_bit, sync)
    else:
        if device_index:
            Set_Zeros(vec, vec_bit, sync)
        else:
            Simple_Multiply(alpha, vec, vec_bit, sync)

    if sync:
        cp.cuda.Device().synchronize()
