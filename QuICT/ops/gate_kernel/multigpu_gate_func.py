import cupy as cp
import numpy as np
import random


__outward_functions = [
    "Simple_Multiply",
    "Device_Prob_Calculator"
]


Simple_Multiply_single = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void SimpleMultiply(complex<float> value, complex<float>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        vec[label] = vec[label]*value;
    }
    ''', 'SimpleMultiply')


Simple_Multiply_double = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void SimpleMultiply(complex<double> value, complex<double>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        vec[label] = vec[label]*value;
    }
    ''', 'SimpleMultiply')


prop_add = cp.ElementwiseKernel(
    'T x, raw T y, int32 index', 'T z',
    'z = (i & index) ? 0 : abs(x) * abs(x)',
    'prop_add')


prob_0 = cp.ElementwiseKernel(
    'T x, raw T y', 'T z',
    'z = 0',
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
            prob = prob_0(vec, vec)
        else:
            prob = prob_1(vec, vec)
    else:
        prob = prop_add(vec, vec, 1 << index)

    prob = MeasureGate_prop_kernel(prob, axis = 0).real

    return prob
