import cupy as cp
import numpy as np
import random


__outward_functions = [
    "Diagonal_Multiply_targ",
    "Diagonal_Multiply_targs",
    "Based_InnerProduct_targ",
    "Based_InnerProduct_targs",
    "Controlled_Multiply_targ",
    "Controlled_Multiply_ctargs",
    "Controlled_Product_ctargs",
    "Controlled_InnerProduct_ctargs",
    "Completed_MxIP_targs",
    "Completed_IPxIP_targs",
    "RDiagonal_Swap_targ",
    "RDiagonal_MultiplySwap_targ",
    "Controlled_MultiplySwap_ctargs",
    "Controlled_Swap_targs",
    "Controlled_Swap_more",
    "Controlled_Multiply_more",
    "Controlled_Swap_tmore",
    "MeasureGate_Apply",
    "ResetGate_Apply",
    "PermGate_Apply",
    "PermFxGate_Apply"
]


Diagonal_Multiply_targ_single_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void Diagonal2x2Multiply(int parg, const complex<float>* mat, complex<float>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        int offset = 1 << parg;

        int _0 = (label >> parg << (parg + 1)) + (label & (offset - 1));
        int _1 = _0 +  offset;

        vec[_0] = vec[_0]*mat[0];
        vec[_1] = vec[_1]*mat[3];
    }
    ''', 'Diagonal2x2Multiply')


Diagonal_Multiply_targ_double_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void Diagonal2x2Multiply(int parg, const complex<double>* mat, complex<double>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        int offset = 1 << parg;

        int _0 = (label >> parg << (parg + 1)) + (label & (offset - 1));
        int _1 = _0 +  offset;

        vec[_0] = vec[_0]*mat[0];
        vec[_1] = vec[_1]*mat[3];
    }
    ''', 'Diagonal2x2Multiply')


Diagonal_Multiply_targs_single_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void Diagonal4x4Multiply(int high, int low, const complex<float>* mat, complex<float>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        const int offset1 = 1 << low;
        const int offset2 = 1 << high;
        const int mask1 = offset1 - 1;
        const int mask2 = offset2 - 1;

        int gw = label >> low << (low + 1);
        int _0 = (gw >> high << (high + 1)) + (gw & (offset2 - offset1)) + (label & mask1);

        int _1 = _0 + offset1;
        int _2 = _0 + offset2;
        int _3 = _1 + offset2;

        vec[_0] = vec[_0]*mat[0];
        vec[_1] = vec[_1]*mat[5];
        vec[_2] = vec[_2]*mat[10];
        vec[_3] = vec[_3]*mat[15];
    }
    ''', 'Diagonal4x4Multiply')


Diagonal_Multiply_targs_double_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void Diagonal4x4Multiply(int high, int low, const complex<double>* mat, complex<double>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        const int offset1 = 1 << low;
        const int offset2 = 1 << high;
        const int mask1 = offset1 - 1;
        const int mask2 = offset2 - 1;

        int gw = label >> low << (low + 1);
        int _0 = (gw >> high << (high + 1)) + (gw & (offset2 - offset1)) + (label & mask1);

        int _1 = _0 + offset1;
        int _2 = _0 + offset2;
        int _3 = _1 + offset2;

        vec[_0] = vec[_0]*mat[0];
        vec[_1] = vec[_1]*mat[5];
        vec[_2] = vec[_2]*mat[10];
        vec[_3] = vec[_3]*mat[15];
    }
    ''', 'Diagonal4x4Multiply')


Based_InnerProduct_targ_single_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void Based2x2InnerProduct(int parg, const complex<float>* mat, complex<float>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        int offset = 1 << parg;

        int _0 = (label >> parg << (parg + 1)) + (label & (offset - 1));
        int _1 = _0 +  offset;

        complex<float> temp_0 = vec[_0];
        vec[_0] = vec[_0]*mat[0] + vec[_1]*mat[1];
        vec[_1] = temp_0*mat[2] + vec[_1]*mat[3];
    }
    ''', 'Based2x2InnerProduct')


Based_InnerProduct_targ_double_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void Based2x2InnerProduct(int parg, const complex<double>* mat, complex<double>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        int offset = 1 << parg;

        int _0 = (label >> parg << (parg + 1)) + (label & (offset - 1));
        int _1 = _0 +  offset;

        complex<double> temp_0 = vec[_0];
        vec[_0] = vec[_0]*mat[0] + vec[_1]*mat[1];
        vec[_1] = temp_0*mat[2] + vec[_1]*mat[3];
    }
    ''', 'Based2x2InnerProduct')


Based_InnerProduct_targs_single_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void Based4x4InnerProduct(int t0, int t1, const complex<float>* mat, complex<float>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        const int offset1 = 1 << t0;
        const int offset2 = 1 << t1;
        const int mask1 = offset1 - 1;
        const int mask2 = offset2 - 1;

        int gw=0, _0=0;

        if (t0 > t1){
            gw = label >> t1 << (t1 + 1);
            _0 = (gw >> t0 << (t0 + 1)) + (gw & (offset1 - offset2)) + (label & mask2);
        }
        else{
            gw = label >> t0 << (t0 + 1);
            _0 = (gw >> t1 << (t1 + 1)) + (gw & (offset2 - offset1)) + (label & mask1);
        }

        int _1 = _0 + offset1;
        int _2 = _0 + offset2;
        int _3 = _1 + offset2;

        complex<float> temp_0 = vec[_0], temp_1 = vec[_1], temp_2 = vec[_2];
        vec[_0] = vec[_0]*mat[0] + vec[_1]*mat[1] + vec[_2]*mat[2] + vec[_3]*mat[3];
        vec[_1] = temp_0*mat[4] + vec[_1]*mat[5] + vec[_2]*mat[6] + vec[_3]*mat[7];
        vec[_2] = temp_0*mat[8] + temp_1*mat[9] + vec[_2]*mat[10] + vec[_3]*mat[11];
        vec[_3] = temp_0*mat[12] + temp_1*mat[13] + temp_2*mat[14] + vec[_3]*mat[15];
    }
    ''', 'Based4x4InnerProduct')


Based_InnerProduct_targs_double_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void Based4x4InnerProduct(int t0, int t1, const complex<double>* mat, complex<double>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        const int offset1 = 1 << t0;
        const int offset2 = 1 << t1;
        const int mask1 = offset1 - 1;
        const int mask2 = offset2 - 1;

        int gw=0, _0=0;

        if (t0 > t1){
            gw = label >> t1 << (t1 + 1);
            _0 = (gw >> t0 << (t0 + 1)) + (gw & (offset1 - offset2)) + (label & mask2);
        }
        else{
            gw = label >> t0 << (t0 + 1);
            _0 = (gw >> t1 << (t1 + 1)) + (gw & (offset2 - offset1)) + (label & mask1);
        }

        int _1 = _0 + offset1;
        int _2 = _0 + offset2;
        int _3 = _1 + offset2;

        complex<double> temp_0 = vec[_0], temp_1 = vec[_1], temp_2 = vec[_2];
        vec[_0] = vec[_0]*mat[0] + vec[_1]*mat[1] + vec[_2]*mat[2] + vec[_3]*mat[3];
        vec[_1] = temp_0*mat[4] + vec[_1]*mat[5] + vec[_2]*mat[6] + vec[_3]*mat[7];
        vec[_2] = temp_0*mat[8] + temp_1*mat[9] + vec[_2]*mat[10] + vec[_3]*mat[11];
        vec[_3] = temp_0*mat[12] + temp_1*mat[13] + temp_2*mat[14] + vec[_3]*mat[15];
    }
    ''', 'Based4x4InnerProduct')


Controlled_Multiply_targ_single_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void Controlled2x2Multiply(int parg, const complex<float> val, complex<float>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        int offset = 1 << parg;

        int _1 = (label >> parg << (parg + 1)) + offset + (label & (offset - 1));

        vec[_1] = vec[_1]*val;
    }
    ''', 'Controlled2x2Multiply')


Controlled_Multiply_targ_double_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void Controlled2x2Multiply(int parg, const complex<double> val, complex<double>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        int offset = 1 << parg;

        int _1 = (label >> parg << (parg + 1)) + offset + (label & (offset - 1));

        vec[_1] = vec[_1]*val;
    }
    ''', 'Controlled2x2Multiply')


Controlled_Multiply_ctargs_single_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void Controlled4x4Multiply(const complex<float>* mat, complex<float>* vec, int c_index, int t_index) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        const int offset_c = 1 << c_index;
        const int offset_t = 1 << t_index;
        const int mask_c = offset_c - 1;
        const int mask_t = offset_t - 1;

        int gw=0, _0=0;

        if (t_index > c_index){
            gw = label >> c_index << (c_index + 1);
            _0 = offset_c + (gw >> t_index << (t_index + 1)) + (gw & (offset_t - offset_c)) + (label & mask_c);
        }
        else
        {
            gw = label >> t_index << (t_index + 1);
            _0 = offset_c + (gw >> c_index << (c_index + 1)) + (gw & (offset_c - offset_t)) + (label & mask_t);
        }

        int _1 = _0 + offset_t;

        vec[_0] = vec[_0]*mat[10];
        vec[_1] = vec[_1]*mat[15];
    }
    ''', 'Controlled4x4Multiply')


Controlled_Multiply_ctargs_double_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void Controlled4x4Multiply(const complex<double>* mat, complex<double>* vec, int c_index, int t_index){
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        const int offset_c = 1 << c_index;
        const int offset_t = 1 << t_index;
        const int mask_c = offset_c - 1;
        const int mask_t = offset_t - 1;

        int gw=0, _0=0;

        if (t_index > c_index){
            gw = label >> c_index << (c_index + 1);
            _0 = offset_c + (gw >> t_index << (t_index + 1)) + (gw & (offset_t - offset_c)) + (label & mask_c);
        }
        else
        {
            gw = label >> t_index << (t_index + 1);
            _0 = offset_c + (gw >> c_index << (c_index + 1)) + (gw & (offset_c - offset_t)) + (label & mask_t);
        }

        int _1 = _0 + offset_t;

        vec[_0] = vec[_0]*mat[10];
        vec[_1] = vec[_1]*mat[15];
    }
    ''', 'Controlled4x4Multiply')


Controlled_Product_ctargs_single_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void Controlled4x4Product(const complex<float> val, complex<float>* vec, int c_index, int t_index) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        const int offset_c = 1 << c_index;
        const int offset_t = 1 << t_index;
        const int mask_c = offset_c - 1;
        const int mask_t = offset_t - 1;

        int gw=0, _0=0;

        if (t_index > c_index){
            gw = label >> c_index << (c_index + 1);
            _0 = offset_c + (gw >> t_index << (t_index + 1)) + (gw & (offset_t - offset_c)) + (label & mask_c);
        }
        else
        {
            gw = label >> t_index << (t_index + 1);
            _0 = offset_c + (gw >> c_index << (c_index + 1)) + (gw & (offset_c - offset_t)) + (label & mask_t);
        }

        _0 = _0 + offset_t;
        vec[_0] = vec[_0]*val;
    }
    ''', 'Controlled4x4Product')


Controlled_Product_ctargs_double_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void Controlled4x4Product(const complex<double> val, complex<double>* vec, int c_index, int t_index){
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        const int offset_c = 1 << c_index;
        const int offset_t = 1 << t_index;
        const int mask_c = offset_c - 1;
        const int mask_t = offset_t - 1;

        int gw=0, _0=0;

        if (t_index > c_index){
            gw = label >> c_index << (c_index + 1);
            _0 = offset_c + (gw >> t_index << (t_index + 1)) + (gw & (offset_t - offset_c)) + (label & mask_c);
        }
        else
        {
            gw = label >> t_index << (t_index + 1);
            _0 = offset_c + (gw >> c_index << (c_index + 1)) + (gw & (offset_c - offset_t)) + (label & mask_t);
        }

        _0 = _0 + offset_t;
        vec[_0] = vec[_0]*val;
    }
    ''', 'Controlled4x4Product')


Controlled_InnerProduct_ctargs_single_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void Controlled4x4InnerProduct(const complex<float>* mat, complex<float>* vec, int c_index, int t_index) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        const int offset_c = 1 << c_index;
        const int offset_t = 1 << t_index;
        const int mask_c = offset_c - 1;
        const int mask_t = offset_t - 1;

        int gw=0, _0=0;

        if (t_index > c_index){
            gw = label >> c_index << (c_index + 1);
            _0 = offset_c + (gw >> t_index << (t_index + 1)) + (gw & (offset_t - offset_c)) + (label & mask_c);
        }
        else
        {
            gw = label >> t_index << (t_index + 1);
            _0 = offset_c + (gw >> c_index << (c_index + 1)) + (gw & (offset_c - offset_t)) + (label & mask_t);
        }

        int _1 = _0 + offset_t;

        complex<float> temp_0 = vec[_0];
        vec[_0] = vec[_0]*mat[10] + vec[_1]*mat[11];
        vec[_1] = temp_0*mat[14] + vec[_1]*mat[15];
    }
    ''', 'Controlled4x4InnerProduct')


Controlled_InnerProduct_ctargs_double_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void Controlled4x4InnerProduct(const complex<double>* mat, complex<double>* vec, int c_index, int t_index) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        const int offset_c = 1 << c_index;
        const int offset_t = 1 << t_index;
        const int mask_c = offset_c - 1;
        const int mask_t = offset_t - 1;

        int gw=0, _0=0;

        if (t_index > c_index){
            gw = label >> c_index << (c_index + 1);
            _0 = offset_c + (gw >> t_index << (t_index + 1)) + (gw & (offset_t - offset_c)) + (label & mask_c);
        }
        else
        {
            gw = label >> t_index << (t_index + 1);
            _0 = offset_c + (gw >> c_index << (c_index + 1)) + (gw & (offset_c - offset_t)) + (label & mask_t);
        }

        int _1 = _0 + offset_t;

        complex<double> temp_0 = vec[_0];
        vec[_0] = vec[_0]*mat[10] + vec[_1]*mat[11];
        vec[_1] = temp_0*mat[14] + vec[_1]*mat[15];
    }
    ''', 'Controlled4x4InnerProduct')


Controlled_MultiplySwap_ctargs_single_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void Controlled4x4MultiSwap(const complex<float>* mat, complex<float>* vec, int c_index, int t_index) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        const int offset_c = 1 << c_index;
        const int offset_t = 1 << t_index;
        const int mask_c = offset_c - 1;
        const int mask_t = offset_t - 1;

        int gw=0, _0=0;

        if (t_index > c_index){
            gw = label >> c_index << (c_index + 1);
            _0 = offset_c + (gw >> t_index << (t_index + 1)) + (gw & (offset_t - offset_c)) + (label & mask_c);
        }
        else
        {
            gw = label >> t_index << (t_index + 1);
            _0 = offset_c + (gw >> c_index << (c_index + 1)) + (gw & (offset_c - offset_t)) + (label & mask_t);
        }

        int _1 = _0 + offset_t;

        complex<float> temp_0 = vec[_0];
        vec[_0] = vec[_1]*mat[11];
        vec[_1] = temp_0*mat[14];
    }
    ''', 'Controlled4x4MultiSwap')


Controlled_MultiplySwap_ctargs_double_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void Controlled4x4MultiSwap(const complex<double>* mat, complex<double>* vec, int c_index, int t_index) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        const int offset_c = 1 << c_index;
        const int offset_t = 1 << t_index;
        const int mask_c = offset_c - 1;
        const int mask_t = offset_t - 1;

        int gw=0, _0=0;

        if (t_index > c_index){
            gw = label >> c_index << (c_index + 1);
            _0 = offset_c + (gw >> t_index << (t_index + 1)) + (gw & (offset_t - offset_c)) + (label & mask_c);
        }
        else
        {
            gw = label >> t_index << (t_index + 1);
            _0 = offset_c + (gw >> c_index << (c_index + 1)) + (gw & (offset_c - offset_t)) + (label & mask_t);
        }

        int _1 = _0 + offset_t;

        complex<double> temp_0 = vec[_0];
        vec[_0] = vec[_1]*mat[11];
        vec[_1] = temp_0*mat[14];
    }
    ''', 'Controlled4x4MultiSwap')


Controlled_Swap_targs_single_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void Controlled4x4Swap(int high, int low, complex<float>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        const int offset1 = 1 << low;
        const int offset2 = 1 << high;
        const int mask1 = offset1 - 1;
        const int mask2 = offset2 - 1;

        int gw = label >> low << (low + 1);
        int _0 = (gw >> high << (high + 1)) + (gw & (offset2 - offset1)) + (label & mask1);

        int _1 = _0 + offset1;
        int _2 = _0 + offset2;

        complex<float> temp_0 = vec[_1];
        vec[_1] = vec[_2];
        vec[_2] = temp_0;
    }
    ''', 'Controlled4x4Swap')


Controlled_Swap_targs_double_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void Controlled4x4Swap(int high, int low, complex<double>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        const int offset1 = 1 << low;
        const int offset2 = 1 << high;
        const int mask1 = offset1 - 1;
        const int mask2 = offset2 - 1;

        int gw = label >> low << (low + 1);
        int _0 = (gw >> high << (high + 1)) + (gw & (offset2 - offset1)) + (label & mask1);

        int _1 = _0 + offset1;
        int _2 = _0 + offset2;

        complex<double> temp_0 = vec[_1];
        vec[_1] = vec[_2];
        vec[_2] = temp_0;
    }
    ''', 'Controlled4x4Swap')


Completed_MxIP_targs_single_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void CompletedMxIP(int high, int low, const complex<float>* mat, complex<float>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        const int offset1 = 1 << low;
        const int offset2 = 1 << high;
        const int mask1 = offset1 - 1;
        const int mask2 = offset2 - 1;

        int gw = label >> low << (low + 1);
        int _0 = (gw >> high << (high + 1)) + (gw & (offset2 - offset1)) + (label & mask1);

        int _1 = _0 + offset1;
        int _2 = _0 + offset2;
        int _3 = _2 + offset1;

        vec[_0] = vec[_0]*mat[0];
        vec[_3] = vec[_3]*mat[15];

        complex<float> temp_0 = vec[_1];
        vec[_1] = vec[_1]*mat[5] + vec[_2]*mat[6];
        vec[_2] = temp_0*mat[9] + vec[_2]*mat[10];
    }
    ''', 'CompletedMxIP')


Completed_MxIP_targs_double_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void CompletedMxIP(int high, int low, const complex<double>* mat, complex<double>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        const int offset1 = 1 << low;
        const int offset2 = 1 << high;
        const int mask1 = offset1 - 1;
        const int mask2 = offset2 - 1;

        int gw = label >> low << (low + 1);
        int _0 = (gw >> high << (high + 1)) + (gw & (offset2 - offset1)) + (label & mask1);

        int _1 = _0 + offset1;
        int _2 = _0 + offset2;
        int _3 = _2 + offset1;

        vec[_0] = vec[_0]*mat[0];
        vec[_3] = vec[_3]*mat[15];

        complex<double> temp_0 = vec[_1];
        vec[_1] = vec[_1]*mat[5] + vec[_2]*mat[6];
        vec[_2] = temp_0*mat[9] + vec[_2]*mat[10];
    }
    ''', 'CompletedMxIP')


Completed_IPxIP_targs_single_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void CompletedIPxIP(int high, int low, const complex<float>* mat, complex<float>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        const int offset1 = 1 << low;
        const int offset2 = 1 << high;
        const int mask1 = offset1 - 1;
        const int mask2 = offset2 - 1;

        int gw = label >> low << (low + 1);
        int _0 = (gw >> high << (high + 1)) + (gw & (offset2 - offset1)) + (label & mask1);

        int _1 = _0 + offset1;
        int _2 = _0 + offset2;
        int _3 = _2 + offset1;

        complex<float> temp_0 = vec[_0];
        vec[_0] = vec[_0]*mat[0] + vec[_3]*mat[3];
        vec[_3] = temp_0*mat[12] + vec[_3]*mat[15];

        complex<float> temp_1 = vec[_1];
        vec[_1] = vec[_1]*mat[5] + vec[_2]*mat[6];
        vec[_2] = temp_1*mat[9] + vec[_2]*mat[10];
    }
    ''', 'CompletedIPxIP')


Completed_IPxIP_targs_double_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void CompletedIPxIP(int high, int low, const complex<double>* mat, complex<double>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        const int offset1 = 1 << low;
        const int offset2 = 1 << high;
        const int mask1 = offset1 - 1;
        const int mask2 = offset2 - 1;

        int gw = label >> low << (low + 1);
        int _0 = (gw >> high << (high + 1)) + (gw & (offset2 - offset1)) + (label & mask1);

        int _1 = _0 + offset1;
        int _2 = _0 + offset2;
        int _3 = _2 + offset1;

        complex<double> temp_0 = vec[_0];
        vec[_0] = vec[_0]*mat[0] + vec[_3]*mat[3];
        vec[_3] = temp_0*mat[12] + vec[_3]*mat[15];

        complex<double> temp_1 = vec[_1];
        vec[_1] = vec[_1]*mat[5] + vec[_2]*mat[6];
        vec[_2] = temp_1*mat[9] + vec[_2]*mat[10];
    }
    ''', 'CompletedIPxIP')


RDiagonal_Swap_targ_single_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void RDiag2x2Swap(int parg, complex<float>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        int offset = 1 << parg;

        int _0 = (label >> parg << (parg + 1)) + (label & (offset - 1));
        int _1 = _0 +  offset;

        complex<float> temp_0 = vec[_0];
        vec[_0] = vec[_1];
        vec[_1] = temp_0;
    }
    ''', 'RDiag2x2Swap')


RDiagonal_Swap_targ_double_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void RDiag2x2Swap(int parg, complex<double>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        int offset = 1 << parg;

        int _0 = (label >> parg << (parg + 1)) + (label & (offset - 1));
        int _1 = _0 +  offset;

        complex<double> temp_0 = vec[_0];
        vec[_0] = vec[_1];
        vec[_1] = temp_0;
    }
    ''', 'RDiag2x2Swap')


RDiagonal_MultiplySwap_targ_single_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void RDiag2x2MultiSwap(int parg, const complex<float>* mat, complex<float>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        int offset = 1 << parg;

        int _0 = (label >> parg << (parg + 1)) + (label & (offset - 1));
        int _1 = _0 +  offset;

        complex<float> temp_0 = vec[_0];
        vec[_0] = vec[_1]*mat[1];
        vec[_1] = temp_0*mat[2];
    }
    ''', 'RDiag2x2MultiSwap')


RDiagonal_MultiplySwap_targ_double_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void RDiag2x2MultiSwap(int parg, const complex<double>* mat, complex<double>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        int offset = 1 << parg;

        int _0 = (label >> parg << (parg + 1)) + (label & (offset - 1));
        int _1 = _0 +  offset;

        complex<double> temp_0 = vec[_0];
        vec[_0] = vec[_1]*mat[1];
        vec[_1] = temp_0*mat[2];
    }
    ''', 'RDiag2x2MultiSwap')


Controlled_Swap_more_single_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void Controlled8x8Swap(int high, int low, int t_index, complex<float>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        const int offset_c1 = 1 << low;
        const int offset_c2 = 1 << high;
        const int offset_t = 1 << t_index;
        const int maskc1 = offset_c1 - 1;
        const int maskc2 = offset_c2 - 1;
        const int mask_t = offset_t - 1;

        int gw = 0, _0 = 0;

        if (t_index < low){
            gw = label >> t_index << (t_index + 1);
            _0 = offset_c1 + (gw >> low << (low + 1)) + (gw & (offset_c1 - offset_t)) + (label & mask_t);
            _0 = offset_c2 + (_0 >> high << (high + 1)) + (_0 & maskc2);
        }else if(t_index < high){
            gw = label >> low << (low + 1);
            _0 = offset_c1 + (gw >> t_index << (t_index + 1)) + (gw & (offset_t - offset_c1)) + (label & maskc1);
            _0 = offset_c2 + (_0 >> high << (high + 1)) + (_0 & maskc2);
        }else{
            gw = label >> low << (low + 1);
            _0 = offset_c1 + (gw >> high << (high + 1)) + (gw & (offset_c2 - offset_c1)) + (label & maskc1);
            _0 = offset_c2 + (_0 >> t_index << (t_index + 1)) + (_0 & mask_t);
        }

        int _1 = _0 + offset_t;

        complex<float> temp_0 = vec[_0];
        vec[_0] = vec[_1];
        vec[_1] = temp_0;
    }
    ''', 'Controlled8x8Swap')


Controlled_Swap_more_double_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void Controlled8x8Swap(int high, int low, int t_index, complex<double>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        const int offset_c1 = 1 << low;
        const int offset_c2 = 1 << high;
        const int offset_t = 1 << t_index;
        const int maskc1 = offset_c1 - 1;
        const int maskc2 = offset_c2 - 1;
        const int mask_t = offset_t - 1;

        int gw = 0, _0 = 0;

        if (t_index < low){
            gw = label >> t_index << (t_index + 1);
            _0 = offset_c1 + (gw >> low << (low + 1)) + (gw & (offset_c1 - offset_t)) + (label & mask_t);
            _0 = offset_c2 + (_0 >> high << (high + 1)) + (_0 & maskc2);
        }else if(t_index < high){
            gw = label >> low << (low + 1);
            _0 = offset_c1 + (gw >> t_index << (t_index + 1)) + (gw & (offset_t - offset_c1)) + (label & maskc1);
            _0 = offset_c2 + (_0 >> high << (high + 1)) + (_0 & maskc2);
        }else{
            gw = label >> low << (low + 1);
            _0 = offset_c1 + (gw >> high << (high + 1)) + (gw & (offset_c2 - offset_c1)) + (label & maskc1);
            _0 = offset_c2 + (_0 >> t_index << (t_index + 1)) + (_0 & mask_t);
        }

        int _1 = _0 + offset_t;

        complex<double> temp_0 = vec[_0];
        vec[_0] = vec[_1];
        vec[_1] = temp_0;
    }
    ''', 'Controlled8x8Swap')


Controlled_Multiply_more_single_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void Controlled8x8Multiply(int high, int low, int t_index, const complex<float>* mat, complex<float>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        const int offset_c1 = 1 << low;
        const int offset_c2 = 1 << high;
        const int offset_t = 1 << t_index;
        const int mask1 = offset_c1 - 1;
        const int mask2 = offset_c2 - 1;
        const int mask_t = offset_t - 1;

        int gw = 0, _0 = 0;

        if (t_index < low){
            gw = label >> t_index << (t_index + 1);
            _0 = offset_c1 + (gw >> low << (low + 1)) + (gw & (offset_c1 - offset_t)) + (label & mask_t);
            _0 = offset_c2 + (_0 >> high << (high + 1)) + (_0 & mask2);
        }else if(t_index < high){
            gw = label >> low << (low + 1);
            _0 = offset_c1 + (gw >> t_index << (t_index + 1)) + (gw & (offset_t - offset_c1)) + (label & mask1);
            _0 = offset_c2 + (_0 >> high << (high + 1)) + (_0 & mask2);
        }else{
            gw = label >> low << (low + 1);
            _0 = offset_c1 + (gw >> high << (high + 1)) + (gw & (offset_c2 - offset_c1)) + (label & mask1);
            _0 = offset_c2 + (_0 >> t_index << (t_index + 1)) + (_0 & mask_t);
        }

        int _1 = _0 + offset_t;

        vec[_0] = vec[_0]*mat[54];
        vec[_1] = vec[_1]*mat[63];
    }
    ''', 'Controlled8x8Multiply')


Controlled_Multiply_more_double_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void Controlled8x8Multiply(int high, int low, int t_index, const complex<double>* mat, complex<double>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        const int offset_c1 = 1 << low;
        const int offset_c2 = 1 << high;
        const int offset_t = 1 << t_index;
        const int mask1 = offset_c1 - 1;
        const int mask2 = offset_c2 - 1;
        const int mask_t = offset_t - 1;

        int gw = 0, _0 = 0;

        if (t_index < low){
            gw = label >> t_index << (t_index + 1);
            _0 = offset_c1 + (gw >> low << (low + 1)) + (gw & (offset_c1 - offset_t)) + (label & mask_t);
            _0 = offset_c2 + (_0 >> high << (high + 1)) + (_0 & mask2);
        }else if(t_index < high){
            gw = label >> low << (low + 1);
            _0 = offset_c1 + (gw >> t_index << (t_index + 1)) + (gw & (offset_t - offset_c1)) + (label & mask1);
            _0 = offset_c2 + (_0 >> high << (high + 1)) + (_0 & mask2);
        }else{
            gw = label >> low << (low + 1);
            _0 = offset_c1 + (gw >> high << (high + 1)) + (gw & (offset_c2 - offset_c1)) + (label & mask1);
            _0 = offset_c2 + (_0 >> t_index << (t_index + 1)) + (_0 & mask_t);
        }

        int _1 = _0 + offset_t;

        vec[_0] = vec[_0]*mat[54];
        vec[_1] = vec[_1]*mat[63];
    }
    ''', 'Controlled8x8Multiply')


Controlled_Swap_tmore_single_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void Controlled8x8Swapt(int high, int low, int c_index, complex<float>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        const int offset_t1 = 1 << low;
        const int offset_t2 = 1 << high;
        const int offset_c = 1 << c_index;
        const int mask1 = offset_t1 - 1;
        const int mask2 = offset_t2 - 1;
        const int mask_c = offset_c - 1;

        int gw = 0, _0 = 0;

        if (c_index < low){
            gw = label >> c_index << (c_index + 1);
            _0 = offset_c + (gw >> low << (low + 1)) + (gw & (offset_t1 - offset_c)) + (label & mask_c);
            _0 = (_0 >> high << (high + 1)) + (_0 & mask2);
        }else if(c_index < high){
            gw = label >> low << (low + 1);
            _0 = offset_c + (gw >> c_index << (c_index + 1)) + (gw & (offset_c - offset_t1)) + (label & mask1);
            _0 = (_0 >> high << (high + 1)) + (_0 & mask2);
        }else{
            gw = label >> low << (low + 1);
            _0 = (gw >> high << (high + 1)) + (gw & (offset_t2 - offset_t1)) + (label & mask1);
            _0 = offset_c + (_0 >> c_index << (c_index + 1)) + (_0 & mask_c);
        }

        int _1 = _0 + offset_t1;
        int _2 = _0 + offset_t2;

        complex<float> temp_0 = vec[_1];
        vec[_1] = vec[_2];
        vec[_2] = temp_0;
    }
    ''', 'Controlled8x8Swapt')


Controlled_Swap_tmore_double_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void Controlled8x8Swapt(int high, int low, int c_index, complex<double>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        const int offset_t1 = 1 << low;
        const int offset_t2 = 1 << high;
        const int offset_c = 1 << c_index;
        const int mask1 = offset_t1 - 1;
        const int mask2 = offset_t2 - 1;
        const int mask_c = offset_c - 1;

        int gw = 0, _0 = 0;

        if (c_index < low){
            gw = label >> c_index << (c_index + 1);
            _0 = offset_c + (gw >> low << (low + 1)) + (gw & (offset_t1 - offset_c)) + (label & mask_c);
            _0 = (_0 >> high << (high + 1)) + (_0 & mask2);
        }else if(c_index < high){
            gw = label >> low << (low + 1);
            _0 = offset_c + (gw >> c_index << (c_index + 1)) + (gw & (offset_c - offset_t1)) + (label & mask1);
            _0 = (_0 >> high << (high + 1)) + (_0 & mask2);
        }else{
            gw = label >> low << (low + 1);
            _0 = (gw >> high << (high + 1)) + (gw & (offset_t2 - offset_t1)) + (label & mask1);
            _0 = offset_c + (_0 >> c_index << (c_index + 1)) + (_0 & mask_c);
        }

        int _1 = _0 + offset_t1;
        int _2 = _0 + offset_t2;

        complex<double> temp_0 = vec[_1];
        vec[_1] = vec[_2];
        vec[_2] = temp_0;
    }
    ''', 'Controlled8x8Swapt')


def Diagonal_Multiply_targ(t_index, mat, vec, vec_bit, sync: bool = False):
    """
    Diagonal matrix (2x2) dot vector
        [[a, 0],    *   vec
         [0, b]]
    """
    task_number = 1 << (vec_bit - 1)
    thread_per_block = min(256, task_number)
    block_num = task_number // thread_per_block

    if vec.dtype == np.complex64:
        Diagonal_Multiply_targ_single_kernel(
            (block_num,),
            (thread_per_block,),
            (t_index, mat, vec)
        )
    else:
        Diagonal_Multiply_targ_double_kernel(
            (block_num,),
            (thread_per_block,),
            (t_index, mat, vec)
        )

    if sync:
        cp.cuda.Device().synchronize()


def Diagonal_Multiply_targs(t_indexes, mat, vec, vec_bit, sync: bool = False):
    """
    Diagonal matrix (4x4) dot vector
        [[a, 0, 0, 0],    *   vec
         [0, b, 0, 0],
         [0, 0, c, 0],
         [0, 0, 0, d]]
    """
    task_number = 1 << (vec_bit - 2)
    thread_per_block = min(256, task_number)
    block_num = task_number // thread_per_block

    if t_indexes[0] > t_indexes[1]:
        high, low = t_indexes[0], t_indexes[1]
    else:
        high, low = t_indexes[1], t_indexes[0]

    if vec.dtype == np.complex64:
        Diagonal_Multiply_targs_single_kernel(
            (block_num,),
            (thread_per_block,),
            (high, low, mat, vec)
        )
    else:
        Diagonal_Multiply_targs_double_kernel(
            (block_num,),
            (thread_per_block,),
            (high, low, mat, vec)
        )

    if sync:
        cp.cuda.Device().synchronize()


def Based_InnerProduct_targ(t_index, mat, vec, vec_bit, sync: bool = False):
    """
    Based matrix (2x2) dot vector
        [[a, b],    *   vec
         [c, d]]
    """
    task_number = 1 << (vec_bit - 1)
    thread_per_block = min(256, task_number)
    block_num = task_number // thread_per_block

    if vec.dtype == np.complex64:
        Based_InnerProduct_targ_single_kernel(
            (block_num,),
            (thread_per_block,),
            (t_index, mat, vec)
        )
    else:
        Based_InnerProduct_targ_double_kernel(
            (block_num,),
            (thread_per_block,),
            (t_index, mat, vec)
        )

    if sync:
        cp.cuda.Device().synchronize()


def Based_InnerProduct_targs(t_indexes, mat, vec, vec_bit, sync: bool = False):
    """
    Based matrix (4x4) dot vector
        [[a, b, c, d],    *   vec
         [e, f, g, h],
         [i, j, k, l],
         [m, n, o, p]]
    """
    task_number = 1 << (vec_bit - 2)
    thread_per_block = min(256, task_number)
    block_num = task_number // thread_per_block

    if vec.dtype == np.complex64:
        Based_InnerProduct_targs_single_kernel(
            (block_num,),
            (thread_per_block,),
            (t_indexes[0], t_indexes[1], mat, vec)
        )
    else:
        Based_InnerProduct_targs_double_kernel(
            (block_num,),
            (thread_per_block,),
            (t_indexes[0], t_indexes[1], mat, vec)
        )

    if sync:
        cp.cuda.Device().synchronize()


def Controlled_Multiply_targ(t_index, val, vec, vec_bit, sync: bool = False):
    """
    Controlled matrix (2x2) dot vector
        [[1, 0],    *   vec
         [0, a]]
    """
    task_number = 1 << (vec_bit - 1)
    thread_per_block = min(256, task_number)
    block_num = task_number // thread_per_block

    if vec.dtype == np.complex64:
        Controlled_Multiply_targ_single_kernel(
            (block_num,),
            (thread_per_block,),
            (t_index, val, vec)
        )
    else:
        Controlled_Multiply_targ_double_kernel(
            (block_num,),
            (thread_per_block,),
            (t_index, val, vec)
        )

    if sync:
        cp.cuda.Device().synchronize()


def Controlled_Multiply_ctargs(c_index, t_index, mat, vec, vec_bit, sync: bool = False):
    """
    Controlled matrix (4x4) dot vector
     e.g.   [[1, 0, 0, 0],    *   vec
             [0, 1, 0, 0],
             [0, 0, a, 0],
             [0, 0, 0, b]]
    """
    task_number = 1 << (vec_bit - 2)
    thread_per_block = min(256, task_number)
    block_num = task_number // thread_per_block

    if vec.dtype == np.complex64:
        Controlled_Multiply_ctargs_single_kernel(
            (block_num,),
            (thread_per_block,),
            (mat, vec, c_index, t_index)
        )
    else:
        Controlled_Multiply_ctargs_double_kernel(
            (block_num,),
            (thread_per_block,),
            (mat, vec, c_index, t_index)
        )

    if sync:
        cp.cuda.Device().synchronize()


def Controlled_Product_ctargs(c_index, t_index, value, vec, vec_bit, sync: bool = False):
    """
    Controlled matrix (4x4) dot vector
     e.g.   [[1, 0, 0, 0],    *   vec
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, a]]
    """
    task_number = 1 << (vec_bit - 2)
    thread_per_block = min(256, task_number)
    block_num = task_number // thread_per_block

    if vec.dtype == np.complex64:
        Controlled_Product_ctargs_single_kernel(
            (block_num,),
            (thread_per_block,),
            (value, vec, c_index, t_index)
        )
    else:
        Controlled_Product_ctargs_double_kernel(
            (block_num,),
            (thread_per_block,),
            (value, vec, c_index, t_index)
        )

    if sync:
        cp.cuda.Device().synchronize()


def Controlled_InnerProduct_ctargs(c_index, t_index, mat, vec, vec_bit, sync: bool = False):
    """
    Controlled matrix (4x4) dot vector
     e.g.   [[1, 0, 0, 0],    *   vec
             [0, 1, 0, 0],
             [0, 0, a, b],
             [0, 0, c, d]]
    """
    task_number = 1 << (vec_bit - 2)
    thread_per_block = min(256, task_number)
    block_num = task_number // thread_per_block

    if vec.dtype == np.complex64:
        Controlled_InnerProduct_ctargs_single_kernel(
            (block_num,),
            (thread_per_block,),
            (mat, vec, c_index, t_index)
        )
    else:
        Controlled_InnerProduct_ctargs_double_kernel(
            (block_num,),
            (thread_per_block,),
            (mat, vec, c_index, t_index)
        )

    if sync:
        cp.cuda.Device().synchronize()


def Completed_MxIP_targs(t_indexes, mat, vec, vec_bit, sync: bool = False):
    """
    Completed matrix (4x4) dot vector
            [[A, 0, 0, 0],    *   vec
             [0, c, d, 0],
             [0, e, f, 0],
             [0, 0, 0, B]]
    """
    task_number = 1 << (vec_bit - 2)
    thread_per_block = min(256, task_number)
    block_num = task_number // thread_per_block

    if t_indexes[0] > t_indexes[1]:
        high, low = t_indexes[0], t_indexes[1]
    else:
        high, low = t_indexes[1], t_indexes[0]

    if vec.dtype == np.complex64:
        Completed_MxIP_targs_single_kernel(
            (block_num,),
            (thread_per_block,),
            (high, low, mat, vec)
        )
    else:
        Completed_MxIP_targs_double_kernel(
            (block_num,),
            (thread_per_block,),
            (high, low, mat, vec)
        )

    if sync:
        cp.cuda.Device().synchronize()


def Completed_IPxIP_targs(t_indexes, mat, vec, vec_bit, sync: bool = False):
    """
    Completed matrix (4x4) dot vector
            [[A, 0, 0, B],    *   vec
             [0, a, b, 0],
             [0, c, d, 0],
             [C, 0, 0, D]]
    """
    task_number = 1 << (vec_bit - 2)
    thread_per_block = min(256, task_number)
    block_num = task_number // thread_per_block

    if t_indexes[0] > t_indexes[1]:
        high, low = t_indexes[0], t_indexes[1]
    else:
        high, low = t_indexes[1], t_indexes[0]

    if vec.dtype == np.complex64:
        Completed_IPxIP_targs_single_kernel(
            (block_num,),
            (thread_per_block,),
            (high, low, mat, vec)
        )
    else:
        Completed_IPxIP_targs_double_kernel(
            (block_num,),
            (thread_per_block,),
            (high, low, mat, vec)
        )

    if sync:
        cp.cuda.Device().synchronize()


def RDiagonal_Swap_targ(t_index, vec, vec_bit, sync: bool = False):
    """
    reverse diagonal matrix (2x2) dot vector
        [[0, 1],        *       vec
         [1, 0]]
    """
    task_number = 1 << (vec_bit - 1)
    thread_per_block = min(256, task_number)
    block_num = task_number // thread_per_block

    if vec.dtype == np.complex64:
        RDiagonal_Swap_targ_single_kernel(
            (block_num,),
            (thread_per_block,),
            (t_index, vec)
        )
    else:
        RDiagonal_Swap_targ_double_kernel(
            (block_num,),
            (thread_per_block,),
            (t_index, vec)
        )

    if sync:
        cp.cuda.Device().synchronize()


def RDiagonal_MultiplySwap_targ(t_index, mat, vec, vec_bit, sync: bool = False):
    """
    reverse diagonal matrix (2x2) dot vector
        [[0, a],        *       vec
         [b, 0]]
    """
    task_number = 1 << (vec_bit - 1)
    thread_per_block = min(256, task_number)
    block_num = task_number // thread_per_block

    if vec.dtype == np.complex64:
        RDiagonal_MultiplySwap_targ_single_kernel(
            (block_num,),
            (thread_per_block,),
            (t_index, mat, vec)
        )
    else:
        RDiagonal_MultiplySwap_targ_double_kernel(
            (block_num,),
            (thread_per_block,),
            (t_index, mat, vec)
        )

    if sync:
        cp.cuda.Device().synchronize()


def Controlled_MultiplySwap_ctargs(c_index, t_index, mat, vec, vec_bit, sync: bool = False):
    """
    Controlled matrix (4x4) dot vector
     e.g.   [[1, 0, 0, 0],    *   vec
             [0, 1, 0, 0],
             [0, 0, 0, a],
             [0, 0, b, 0]]
    """
    task_number = 1 << (vec_bit - 2)
    thread_per_block = min(256, task_number)
    block_num = task_number // thread_per_block

    if vec.dtype == np.complex64:
        Controlled_MultiplySwap_ctargs_single_kernel(
            (block_num,),
            (thread_per_block,),
            (mat, vec, c_index, t_index)
        )
    else:
        Controlled_MultiplySwap_ctargs_double_kernel(
            (block_num,),
            (thread_per_block,),
            (mat, vec, c_index, t_index)
        )

    if sync:
        cp.cuda.Device().synchronize()


def Controlled_Swap_targs(t_indexes, vec, vec_bit, sync: bool = False):
    """
    Controlled matrix (4x4) dot vector
            [[1, 0, 0, 0],    *   vec
             [0, 0, 1, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1]]
    """
    task_number = 1 << (vec_bit - 2)
    thread_per_block = min(256, task_number)
    block_num = task_number // thread_per_block

    if t_indexes[0] > t_indexes[1]:
        high, low = t_indexes[0], t_indexes[1]
    else:
        high, low = t_indexes[1], t_indexes[0]

    if vec.dtype == np.complex64:
        Controlled_Swap_targs_single_kernel(
            (block_num,),
            (thread_per_block,),
            (high, low, vec)
        )
    else:
        Controlled_Swap_targs_double_kernel(
            (block_num,),
            (thread_per_block,),
            (high, low, vec)
        )

    if sync:
        cp.cuda.Device().synchronize()


def Controlled_Swap_more(c_indexes, t_index, vec, vec_bit, sync: bool = False):
    """
    Controlled matrix (8x8) dot vector
       [[1, 0, 0, 0, 0, 0, 0, 0],       *       vec
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0]]
    """
    task_number = 1 << (vec_bit - 3)
    thread_per_block = min(256, task_number)
    block_num = task_number // thread_per_block

    if c_indexes[0] > c_indexes[1]:
        high, low = c_indexes[0], c_indexes[1]
    else:
        high, low = c_indexes[1], c_indexes[0]

    if vec.dtype == np.complex64:
        Controlled_Swap_more_single_kernel(
            (block_num,),
            (thread_per_block,),
            (high, low, t_index, vec)
        )
    else:
        Controlled_Swap_more_double_kernel(
            (block_num,),
            (thread_per_block,),
            (high, low, t_index, vec)
        )

    if sync:
        cp.cuda.Device().synchronize()


def Controlled_Multiply_more(c_indexes, t_index, mat, vec, vec_bit, sync: bool = False):
    """
    Controlled matrix (8x8) dot vector
       [[1, 0, 0, 0, 0, 0, 0, 0],       *       vec
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, a, 0],
        [0, 0, 0, 0, 0, 0, 0, b]]
    """
    task_number = 1 << (vec_bit - 3)
    thread_per_block = min(256, task_number)
    block_num = task_number // thread_per_block

    if c_indexes[0] > c_indexes[1]:
        high, low = c_indexes[0], c_indexes[1]
    else:
        high, low = c_indexes[1], c_indexes[0]

    if vec.dtype == np.complex64:
        Controlled_Multiply_more_single_kernel(
            (block_num,),
            (thread_per_block,),
            (high, low, t_index, mat, vec)
        )
    else:
        Controlled_Multiply_more_double_kernel(
            (block_num,),
            (thread_per_block,),
            (high, low, t_index, mat, vec)
        )

    if sync:
        cp.cuda.Device().synchronize()


def Controlled_Swap_tmore(t_indexes, c_index, vec, vec_bit, sync: bool = False):
    """
    Controlled matrix (8x8) dot vector
       [[1, 0, 0, 0, 0, 0, 0, 0],       *       vec
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1]]
    """
    task_number = 1 << (vec_bit - 3)
    thread_per_block = min(256, task_number)
    block_num = task_number // thread_per_block

    if t_indexes[0] > t_indexes[1]:
        high, low = t_indexes[0], t_indexes[1]
    else:
        high, low = t_indexes[1], t_indexes[0]

    if vec.dtype == np.complex64:
        Controlled_Swap_tmore_single_kernel(
            (block_num,),
            (thread_per_block,),
            (high, low, c_index, vec)
        )
    else:
        Controlled_Swap_tmore_double_kernel(
            (block_num,),
            (thread_per_block,),
            (high, low, c_index, vec)
        )

    if sync:
        cp.cuda.Device().synchronize()


"""
Special Gates: MeasureGate, ResetGate and PermGate
"""
prop_add = cp.ElementwiseKernel(
    'T x, raw T y, int32 index', 'T z',
    'z = (i & index) ? 0 : abs(x) * abs(x)',
    'prop_add')


MeasureGate_prop = cp.ReductionKernel(
    'T x',
    'T y',
    'x',
    'a + b',
    'y = abs(a)',
    '0',
    'MeasureGate_prop')


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


ResetGate0_single_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void ResetGate0Float(const int index, const float generation, complex<float>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;
        int _0 = (label & ((1 << index) - 1))
                + (label >> index << (index + 1));
        vec[_0] = vec[_0] / generation;
        vec[_0 + (1 << index)] = complex<float>(0, 0);
    }
    ''', 'ResetGate0Float')


ResetGate1_single_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void ResetGate1Float(const int index, const float generation, complex<float>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;
        int _0 = (label & ((1 << index) - 1))
                + (label >> index << (index + 1));
        int _1 = _0 + (1 << index);

        vec[_0] = vec[_1];
        vec[_1] = complex<float>(0, 0);
    }
    ''', 'ResetGate1Float')


ResetGate0_double_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void ResetGate0Double(const int index, const double generation, complex<double>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;
        int _0 = (label & ((1 << index) - 1))
                + (label >> index << (index + 1));
        vec[_0] = vec[_0] / generation;
        vec[_0 + (1 << index)] = complex<double>(0, 0);
    }
    ''', 'ResetGate0Double')


ResetGate1_double_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void ResetGate1Double(const int index, const double generation, complex<double>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;
        int _1 = (label & ((1 << index) - 1))
                + (label >> index << (index + 1))
                + (1 << index);

        vec[label] = vec[_1];
    }
    ''', 'ResetGate1Double')


def MeasureGate_Apply(index, vec, vec_bit, sync: bool = False, multigpu_prob=None):
    """
    Measure Gate Measure.
    """
    if multigpu_prob is None:
        prob = prop_add(vec, vec, 1 << index)
        prob = MeasureGate_prop(prob, axis=0).real
        prob = prob.get()
    else:
        prob = multigpu_prob

    _0 = random.random()
    _1 = _0 > prob

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


def ResetGate_Apply(index, vec, vec_bit, sync: bool = False, multigpu_prob=None):
    """
    Measure Gate Measure.
    """
    if not multigpu_prob:
        prob = prop_add(vec, vec, 1 << index)
        prob = MeasureGate_prop(prob, axis=0).real
        prob = prob.get()
    else:
        prob = multigpu_prob

    task_number = 1 << (vec_bit - 1)
    thread_per_block = min(256, task_number)
    block_num = task_number // thread_per_block

    alpha = np.float64(np.sqrt(prob))

    if alpha < 1e-6:
        if vec.dtype == np.complex64:
            ResetGate1_single_kernel(
                (block_num, ),
                (thread_per_block,),
                (index, alpha, vec)
            )
        else:
            ResetGate1_double_kernel(
                (block_num,),
                (thread_per_block,),
                (index, alpha, vec)
            )
    else:
        if vec.dtype == np.complex64:
            ResetGate0_single_kernel(
                (block_num,),
                (thread_per_block,),
                (index, alpha, vec)
            )
        else:
            ResetGate0_double_kernel(
                (block_num,),
                (thread_per_block,),
                (index, alpha, vec)
            )

    if sync:
        cp.cuda.Device().synchronize()


kernel_funcs = list(locals().keys())
for name in kernel_funcs:
    if name.endswith("kernel"):
        locals()[name].compile()
