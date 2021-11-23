import cupy as cp
import numpy as np

from .gate_function import prop_add, MeasureGate_prop_kernel


__outward_functions = [
    "Simple_Multiply",
    "Float_Multiply",
    "Device_Prob_Calculator"
]


Simple_Multiply_single = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void SimpleMultiply(complex<float> val, complex<float>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        vec[label] = vec[label]*val;
    }
    ''', 'SimpleMultiply')


Simple_Multiply_double = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void SimpleMultiply(complex<double> val, complex<double>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        vec[label] = vec[label]*val;
    }
    ''', 'SimpleMultiply')


Float_Multiply_single = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void FloatMultiply(const float value, complex<float>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        vec[label] = vec[label] * value;
    }
    ''', 'FloatMultiply')


Float_Multiply_double = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void FloatMultiply(const double value, complex<double>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        vec[label] = vec[label] * value;
    }
    ''', 'FloatMultiply')


prob_0 = cp.ElementwiseKernel(
    'T x, raw T y', 'T z',
    'z = 0',
    'prop_add')


prob_1 = cp.ElementwiseKernel(
    'T x, raw T y', 'T z',
    'z = abs(x) * abs(x)',
    'prop_add')


def Simple_Multiply(val, vec, vec_bit, sync: bool = False):
    task_number = 1 << vec_bit
    thread_per_block = min(256, task_number)
    block_num = task_number // thread_per_block

    if vec.dtype == np.complex64:
        Simple_Multiply_single(
            (block_num,),
            (thread_per_block,),
            (val, vec)
        )
    else:
        Simple_Multiply_double(
            (block_num,),
            (thread_per_block,),
            (val, vec)
        )

    if sync:
        cp.cuda.Device().synchronize()


def Device_Prob_Calculator(index, vec, device_qubits, dev_id):
    """
    Measure Gate Measure.
    """
    if index >= device_qubits:
        if dev_id & (1 << (index - device_qubits)):
            prob = prob_0(vec, vec)
        else:
            prob = prob_1(vec, vec)
    else:
        prob = prop_add(vec, vec, 1 << index)

    prob = MeasureGate_prop_kernel(prob, axis=0).real

    return prob


def Float_Multiply(val, vec, vec_bit, sync: bool = False):
    task_number = 1 << vec_bit
    thread_per_block = min(256, task_number)
    block_num = task_number // thread_per_block

    if vec.dtype == np.complex64:
        Float_Multiply_single(
            (block_num,),
            (thread_per_block,),
            (val, vec)
        )
    else:
        Float_Multiply_double(
            (block_num,),
            (thread_per_block,),
            (val, vec)
        )

    if sync:
        cp.cuda.Device().synchronize()
