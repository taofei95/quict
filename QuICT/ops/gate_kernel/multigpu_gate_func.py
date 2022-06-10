import cupy as cp
import numpy as np

from .gate_function import prop_add_double_kernel, prop_add_single_kernel, MeasureGate_prop


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
Simple_Multiply_single.compile()


Simple_Multiply_double = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void SimpleMultiply(complex<double> val, complex<double>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        vec[label] = vec[label]*val;
    }
    ''', 'SimpleMultiply')
Simple_Multiply_double.compile()


Float_Multiply_single = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void FloatMultiply(const float value, complex<float>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        vec[label] = vec[label] * value;
    }
    ''', 'FloatMultiply')
Float_Multiply_single.compile()


Float_Multiply_double = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void FloatMultiply(const double value, complex<double>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        vec[label] = vec[label] * value;
    }
    ''', 'FloatMultiply')
Float_Multiply_double.compile()


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
    kernel_function = prop_add_double_kernel if vec.dtype == np.complex128 else \
        prop_add_single_kernel

    if index >= device_qubits:
        if dev_id & (1 << (index - device_qubits)):
            temp = cp.zeros(1, dtype=np.complex128)
            return temp[0].real

        task_number = vec.size
    else:
        task_number = vec.size // 2

    thread_per_block = min(256, task_number)
    block_num = task_number // thread_per_block
    out = cp.empty(task_number, dtype=np.complex128)
    kernel_function(
        (block_num, ),
        (thread_per_block, ),
        (index, vec, out)
    )

    prob = MeasureGate_prop(out, axis=0).real
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
