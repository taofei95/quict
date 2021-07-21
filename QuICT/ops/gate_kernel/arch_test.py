import cupy as cp
import numpy as np

import random


multiply_1arg_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void Multiply2x2Matrix(int parg, const complex<float>* mat, complex<float>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        int offset = 1 << parg;

        int _0 = (label >> parg << (parg + 1)) + (label & (offset - 1));
        int _1 = _0 +  offset;

        vec[_0] = vec[_0]*mat[0];
        vec[_1] = vec[_1]*mat[3];
    }
    ''', 'Multiply2x2Matrix')


multiply_2args_single_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void MultiplyTwoArgs(int* pargs, const complex<float>* mat, complex<float>* vec, bool reverse) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        const int offset1 = 1 << pargs[0];
        const int offset2 = 1 << pargs[1];
        const int mask1 = offset1 - 1;
        const int mask2 = offset2 - 1;

        int gw = label >> pargs[0] << (pargs[0] + 1);
        int _0 = (gw >> pargs[1] << (pargs[1] + 1)) + (gw & (offset2 - offset1)) + (label & mask1);

        int _1=0, _2=0, _3=0;

        if(reverse){
            _1 = _0 + offset2;
            _2 = _0 + offset1;
            _3 = _2 + offset2;
        }else{
            _1 = _0 + offset1;
            _2 = _0 + offset2;
            _3 = _2 + offset1;
        }


        vec[_0] = vec[_0]*mat[0];
        vec[_1] = vec[_1]*mat[5];
        vec[_2] = vec[_2]*mat[10];
        vec[_3] = vec[_3]*mat[15];
    }
    ''', 'MultiplyTwoArgs')


iproduct_1arg_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void InnerProduct2x2Matrix(int parg, const complex<T>* mat, complex<T>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        int offset = 1 << parg

        int _0 = (label >> parg << (parg + 1)) + (label & (offset - 1));
        int _1 = _0 +  offset;

        complex<T> temp_0 = vec[_0];
        vec[_0] = vec[_0]*mat[0] + vec[_1]*mat[1];
        vec[_1] = temp_0*mat[2] + vec[_1]*mat[3];
    }
    ''', 'InnerProduct2x2Matrix')


iproduct_2args_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void InnerProduct4x4Matrix(int* pargs, const complex<float>* mat, complex<float>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        const int offset1 = 1 << pargs[0];
        const int offset2 = 1 << pargs[1];
        const int mask1 = offset1 - 1;
        const int mask2 = offset2 - 1;

        int gw = label >> pargs[0] << (pargs[0] + 1);
        int _0 = (gw >> pargs[1] << (pargs[1] + 1)) + (gw & (offset2 - offset1)) + (label & mask1);

        int _1 = _0 + offset1;
        int _2 = _0 + offset2;
        int _3 = _1 + offset2;

        complex<float> temp_0 = vec[_0], temp_1 = vec[_1], temp_2 = vec[_2]
        vec[_0] = vec[_0]*mat[0] + vec[_1]*mat[1] + vec[_2]*mat[2] + vec[_3]*mat[3];
        vec[_1] = temp_0*mat[4] + vec[_1]*mat[5] + vec[_2]*mat[6] + vec[_3]*mat[7];
        vec[_2] = temp_0*mat[8] + temp_1*mat[9] + vec[_2]*mat[10] + vec[_3]*mat[11];
        vec[_3] = temp_0*mat[12] + temp_1*mat[13] + temp_2*mat[14] + vec[_3]*mat[15];
    }
    ''', 'InnerProduct2x2Matrix')


pim_1arg_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void pimMultiply2x2Matrix(int parg, const complex<float>* mat, complex<float>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        int _1 = (label >> parg << (parg + 1)) + (1 << parg) + (label & (offset - 1));

        vec[_1] = vec[_1]*mat[3];
    }
    ''', 'pimMultiply2x2Matrix')


pim_multiply_2args_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void pimMultiply4x4Matrix(const complex<float>* mat, complex<float>* vec, int c_index, int t_index, int bit_pos) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        const int offset_c = 1 << c_index;
        const int offset_t = 1 << t_index;
        const int mask_c = offset_c - 1;
        const int mask_t = offset_t - 1;

        int gw=0, _0=0;

        if (t_index > c_index){
            int gw = label >> c_index << (c_index + 1);
            int _0 = (gw >> t_index << (t_index + 1)) + (gw & (offset_t - offset_c)) + (label & mask_c);
        }
        else
        {
            int gw = label >> t_index << (t_index + 1);
            int _0 = (gw >> c_index << (c_index + 1)) + (gw & (offset_c - offset_t)) + (label & mask_t);
        }

        if(bit_pos & 1){
            vec[_0] = vec[_0]*mat[0];
        }
        
        if(bit_pos & 2)
        {
            int _1 = _0 + offset_t;
            vec[_1] = vec[_1]*mat[5];
        }
        
        if(bit_pos & 4)
        {
            int _2 = _0 + offset_c;
            vec[_2] = vec[_2]*mat[10];
        }
        
        if(bit_pos & 8)
        {
            int _3  = _0 + offset_c + offset_t;
            vec[_3] = vec[_3]*mat[15];
        }
    }
    ''', 'pimMultiply4x4Matrix')


pim_innerproduct_2args_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void pimInnerProduct4x4Matrix(const complex<float>* mat, complex<float>* vec, int c_index, int t_index) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        const int offset_c = 1 << c_index;
        const int offset_t = 1 << t_index;
        const int mask_c = offset_c - 1;
        const int mask_t = offset_t - 1;

        int gw=0, _0=0;

        if (t_index > c_index){
            int gw = label >> c_index << (c_index + 1);
            int _0 = offset_c + (gw >> t_index << (t_index + 1)) + (gw & (offset_t - offset_c)) + (label & mask_c);
        }
        else
        {
            int gw = label >> t_index << (t_index + 1);
            int _0 = offset_c + (gw >> c_index << (c_index + 1)) + (gw & (offset_c - offset_t)) + (label & mask_t);
        }

        int _1 = _0 + offset_t;

        complex<T> temp_0 = vec[_0];
        vec[_0] = vec[_0]*mat[10] + vec[_1]*mat[11];
        vec[_1] = temp_0*mat[14] + vec[_1]*mat[15];
    }
    ''', 'pimInnerProduct4x4Matrix')


Completed_MxIP_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void CompletedMxIP(int* pargs, const complex<float>* mat, complex<float>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        const int offset1 = 1 << pargs[0];
        const int offset2 = 1 << pargs[1];
        const int mask1 = offset1 - 1;
        const int mask2 = offset2 - 1;

        int gw = label >> pargs[0] << (pargs[0] + 1);
        int _0 = (gw >> pargs[1] << (pargs[1] + 1)) + (gw & (offset2 - offset1)) + (label & mask1);

        int _1 = _0 + offset1;
        int _2 = _0 + offset2;
        int _3 = _2 + offset1;

        vec[_0] = vec[_0]*mat[0];
        vec[_3] = vec[_3]*mat[15];

        complex<T> temp_0 = vec[_1];
        vec[_1] = vec[_1]*mat[5] + vec[_2]*mat[6];
        vec[_2] = temp_0*mat[9] + vec[_2]*mat[10];
    }
    ''', 'CompletedMxIP')


Completed_IPxIP_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void CompletedIPxIP(int* pargs, const complex<float>* mat, complex<float>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        const int offset1 = 1 << pargs[0];
        const int offset2 = 1 << pargs[1];
        const int mask1 = offset1 - 1;
        const int mask2 = offset2 - 1;

        int gw = label >> pargs[0] << (pargs[0] + 1);
        int _0 = (gw >> pargs[1] << (pargs[1] + 1)) + (gw & (offset2 - offset1)) + (label & mask1);

        int _1 = _0 + offset1;
        int _2 = _0 + offset2;
        int _3 = _2 + offset1;

        complex<T> temp_0 = vec[_0];
        vec[_0] = vec[_0]*mat[0] + vec[_3]*mat[3];
        vec[_3] = temp_0*mat[12] + vec[_3]*mat[15];

        complex<T> temp_1 = vec[_1];
        vec[_1] = vec[_1]*mat[5] + vec[_2]*mat[6];
        vec[_2] = temp_1*mat[9] + vec[_2]*mat[10];
    }
    ''', 'CompletedIPxIP')


swap_1arg_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void Swap2x2Matrix(int parg, const complex<float>* mat, complex<float>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        int offset = 1 << parg;

        int _0 = (label >> parg << (parg + 1)) + (label & (offset - 1));
        int _1 = _0 +  offset;

        complex<T> temp_0 = vec[_0];
        vec[_0] = vec[_1];
        vec[_1] = temp_0;
    }
    ''', 'Swap2x2Matrix')


def multiply_1arg_matrixdot(t_index, mat, vec, vec_bit, sync: bool = False):
    """
    CRzGate dot function.
    """
    task_number = 1 << (vec_bit - 1)
    thread_per_block = min(256, task_number)
    block_num = task_number // thread_per_block

    multiply_1arg_kernel(
        (block_num,),
        (thread_per_block,),
        (t_index, mat, vec)
    )

    if sync:
        cp.cuda.Device().synchronize()


def multiply_2args_matrixdot(c_index, t_index, mat, vec, vec_bit, sync: bool = False):
    """
    CRzGate dot function.
    """
    task_number = 1 << (vec_bit - 2)
    thread_per_block = min(256, task_number)
    block_num = task_number // thread_per_block

    if c_index > t_index:
        indexes = cp.array([t_index, c_index], dtype=np.int32)
        reverse = False
    else:
        indexes = cp.array([c_index, t_index], dtype=np.int32)
        reverse = True

    multiply_2args_single_kernel(
        (block_num,),
        (thread_per_block,),
        (indexes, mat, vec, reverse)
    )

    if sync:
        cp.cuda.Device().synchronize()


def innerproduct_1arg_matrixdot(t_index, mat, vec, vec_bit, sync: bool = False):
    """
    CRzGate dot function.
    """
    task_number = 1 << (vec_bit - 1)
    thread_per_block = min(256, task_number)
    block_num = task_number // thread_per_block

    iproduct_1arg_kernel(
        (block_num,),
        (thread_per_block,),
        (t_index, mat, vec)
    )

    if sync:
        cp.cuda.Device().synchronize()


def innerproduct_2args_matrixdot(c_index, t_index, mat, vec, vec_bit, sync: bool = False):
    """
    CRzGate dot function.
    """
    task_number = 1 << (vec_bit - 2)
    thread_per_block = min(256, task_number)
    block_num = task_number // thread_per_block

    if c_index > t_index:
        indexes = cp.array([t_index, c_index], dtype=np.int32)
    else:
        indexes = cp.array([c_index, t_index], dtype=np.int32)

    iproduct_2arg_kernel(
        (block_num,),
        (thread_per_block,),
        (indexes, mat, vec)
    )

    if sync:
        cp.cuda.Device().synchronize()

