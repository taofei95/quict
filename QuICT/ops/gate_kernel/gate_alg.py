import cupy as cp
import numpy as np


__outward_functions = [
    "Diagonal_Multiply_targ",
    "Diagonal_Multiply_targs",
    "Based_InnerProduct_targ",
    "Based_InnerProduct_targs",
    "PartialIdentity_Multiply_targ",
    "PartialIdentity_Multiply_ctargs",
    "PartialIdentity_InnerProduct_ctargs",
    "Completed_MxIP_targs",
    "Completed_IPxIP_targs",
    "RDiagonal_swap_targ",
    "RDiagonal_MultiplySwap_targ",
    "PartialIdentity_MultiplySwap_ctargs",
    "RDiagonal_swap_targs"
]


Diagonal_Multiply_targ_single = cp.RawKernel(r'''
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


Diagonal_Multiply_targ_double = cp.RawKernel(r'''
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


Diagonal_Multiply_targs_single = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void Diagonal4x4Multiply(int* pargs, const complex<float>* mat, complex<float>* vec) {
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

        vec[_0] = vec[_0]*mat[0];
        vec[_1] = vec[_1]*mat[5];
        vec[_2] = vec[_2]*mat[10];
        vec[_3] = vec[_3]*mat[15];
    }
    ''', 'Diagonal4x4Multiply')


Diagonal_Multiply_targs_double = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void Diagonal4x4Multiply(int* pargs, const complex<double>* mat, complex<double>* vec) {
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

        vec[_0] = vec[_0]*mat[0];
        vec[_1] = vec[_1]*mat[5];
        vec[_2] = vec[_2]*mat[10];
        vec[_3] = vec[_3]*mat[15];
    }
    ''', 'Diagonal4x4Multiply')


Based_InnerProduct_targ_single = cp.RawKernel(r'''
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


Based_InnerProduct_targ_double = cp.RawKernel(r'''
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


Based_InnerProduct_targs_single = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void Based4x4InnerProduct(int* pargs, const complex<float>* mat, complex<float>* vec) {
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
    ''', 'Based4x4InnerProduct')


Based_InnerProduct_targs_double = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void Based4x4InnerProduct(int* pargs, const complex<double>* mat, complex<double>* vec) {
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

        complex<double> temp_0 = vec[_0], temp_1 = vec[_1], temp_2 = vec[_2]
        vec[_0] = vec[_0]*mat[0] + vec[_1]*mat[1] + vec[_2]*mat[2] + vec[_3]*mat[3];
        vec[_1] = temp_0*mat[4] + vec[_1]*mat[5] + vec[_2]*mat[6] + vec[_3]*mat[7];
        vec[_2] = temp_0*mat[8] + temp_1*mat[9] + vec[_2]*mat[10] + vec[_3]*mat[11];
        vec[_3] = temp_0*mat[12] + temp_1*mat[13] + temp_2*mat[14] + vec[_3]*mat[15];
    }
    ''', 'Based4x4InnerProduct')


PartialIdentity_Multiply_targ_single = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void PartI2x2Multiply(int parg, const complex<float>* mat, complex<float>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        int offset = 1 << parg;

        int _1 = (label >> parg << (parg + 1)) + offset + (label & (offset - 1));

        vec[_1] = vec[_1]*mat[3];
    }
    ''', 'PartI2x2Multiply')


PartialIdentity_Multiply_targ_double = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void PartI2x2Multiply(int parg, const complex<double>* mat, complex<double>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        int offset = 1 << parg;

        int _1 = (label >> parg << (parg + 1)) + offset + (label & (offset - 1));

        vec[_1] = vec[_1]*mat[3];
    }
    ''', 'PartI2x2Multiply')


PartialIdentity_Multiply_ctargs_single = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void PartI4x4Multiply(const complex<float>* mat, complex<float>* vec, int c_index, int t_index, int bit_pos) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        const int offset_c = 1 << c_index;
        const int offset_t = 1 << t_index;
        const int mask_c = offset_c - 1;
        const int mask_t = offset_t - 1;

        int gw=0, _0=0;

        if (t_index > c_index){
            gw = label >> c_index << (c_index + 1);
            _0 = (gw >> t_index << (t_index + 1)) + (gw & (offset_t - offset_c)) + (label & mask_c);
        }
        else
        {
            gw = label >> t_index << (t_index + 1);
            _0 = (gw >> c_index << (c_index + 1)) + (gw & (offset_c - offset_t)) + (label & mask_t);
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
    ''', 'PartI4x4Multiply')


PartialIdentity_Multiply_ctargs_double = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void PartI4x4Multiply(const complex<double>* mat, complex<double>* vec, int c_index, int t_index, int bit_pos) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        const int offset_c = 1 << c_index;
        const int offset_t = 1 << t_index;
        const int mask_c = offset_c - 1;
        const int mask_t = offset_t - 1;

        int gw=0, _0=0;

        if (t_index > c_index){
            gw = label >> c_index << (c_index + 1);
            _0 = (gw >> t_index << (t_index + 1)) + (gw & (offset_t - offset_c)) + (label & mask_c);
        }
        else
        {
            gw = label >> t_index << (t_index + 1);
            _0 = (gw >> c_index << (c_index + 1)) + (gw & (offset_c - offset_t)) + (label & mask_t);
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
    ''', 'PartI4x4Multiply')


PartialIdentity_InnerProduct_ctargs_single = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void PartI4x4InnerProduct(const complex<float>* mat, complex<float>* vec, int c_index, int t_index) {
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
    ''', 'PartI4x4InnerProduct')


PartialIdentity_InnerProduct_ctargs_double = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void PartI4x4InnerProduct(const complex<double>* mat, complex<double>* vec, int c_index, int t_index) {
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
    ''', 'PartI4x4InnerProduct')


PartialIdentity_MultiplySwap_ctargs_single = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void PartI4x4MultiSwap(const complex<float>* mat, complex<float>* vec, int c_index, int t_index) {
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
    ''', 'PartI4x4MultiSwap')


PartialIdentity_MultiplySwap_ctargs_double = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void PartI4x4MultiSwap(const complex<double>* mat, complex<double>* vec, int c_index, int t_index) {
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
    ''', 'PartI4x4MultiSwap')


PartialIdentity_swap_targs_single = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void PartI4x4Swap(int* pargs, const complex<float>* mat, complex<float>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        const int offset1 = 1 << pargs[0];
        const int offset2 = 1 << pargs[1];
        const int mask1 = offset1 - 1;
        const int mask2 = offset2 - 1;

        int gw = label >> pargs[0] << (pargs[0] + 1);
        int _0 = (gw >> pargs[1] << (pargs[1] + 1)) + (gw & (offset2 - offset1)) + (label & mask1);

        int _1 = _0 + offset1;
        int _2 = _0 + offset2;

        complex<float> temp_0 = vec[_1];
        vec[_1] = vec[_2];
        vec[_2] = temp_0;
    }
    ''', 'PartI4x4Swap')


PartialIdentity_swap_targs_double = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void PartI4x4Swap(int* pargs, const complex<double>* mat, complex<double>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        const int offset1 = 1 << pargs[0];
        const int offset2 = 1 << pargs[1];
        const int mask1 = offset1 - 1;
        const int mask2 = offset2 - 1;

        int gw = label >> pargs[0] << (pargs[0] + 1);
        int _0 = (gw >> pargs[1] << (pargs[1] + 1)) + (gw & (offset2 - offset1)) + (label & mask1);

        int _1 = _0 + offset1;
        int _2 = _0 + offset2;

        complex<double> temp_0 = vec[_1];
        vec[_1] = vec[_2];
        vec[_2] = temp_0;
    }
    ''', 'PartI4x4Swap')


Completed_MxIP_targs_single = cp.RawKernel(r'''
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

        complex<float> temp_0 = vec[_1];
        vec[_1] = vec[_1]*mat[5] + vec[_2]*mat[6];
        vec[_2] = temp_0*mat[9] + vec[_2]*mat[10];
    }
    ''', 'CompletedMxIP')


Completed_MxIP_targs_double = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void CompletedMxIP(int* pargs, const complex<double>* mat, complex<double>* vec) {
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

        complex<double> temp_0 = vec[_1];
        vec[_1] = vec[_1]*mat[5] + vec[_2]*mat[6];
        vec[_2] = temp_0*mat[9] + vec[_2]*mat[10];
    }
    ''', 'CompletedMxIP')


Completed_IPxIP_targs_single = cp.RawKernel(r'''
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

        complex<float> temp_0 = vec[_0];
        vec[_0] = vec[_0]*mat[0] + vec[_3]*mat[3];
        vec[_3] = temp_0*mat[12] + vec[_3]*mat[15];

        complex<float> temp_1 = vec[_1];
        vec[_1] = vec[_1]*mat[5] + vec[_2]*mat[6];
        vec[_2] = temp_1*mat[9] + vec[_2]*mat[10];
    }
    ''', 'CompletedIPxIP')


Completed_IPxIP_targs_double = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void CompletedIPxIP(int* pargs, const complex<double>* mat, complex<double>* vec) {
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

        complex<double> temp_0 = vec[_0];
        vec[_0] = vec[_0]*mat[0] + vec[_3]*mat[3];
        vec[_3] = temp_0*mat[12] + vec[_3]*mat[15];

        complex<double> temp_1 = vec[_1];
        vec[_1] = vec[_1]*mat[5] + vec[_2]*mat[6];
        vec[_2] = temp_1*mat[9] + vec[_2]*mat[10];
    }
    ''', 'CompletedIPxIP')


RDiagonal_swap_targ_single = cp.RawKernel(r'''
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


RDiagonal_swap_targ_double = cp.RawKernel(r'''
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


RDiagonal_MultiplySwap_targ_single = cp.RawKernel(r'''
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


RDiagonal_MultiplySwap_targ_double = cp.RawKernel(r'''
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


PartialIdentity_swap_more_single = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void PartI8x8Swap(int* pargs, int c_index, const complex<float>* mat, complex<float>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        const int offset1 = 1 << pargs[0];
        const int offset2 = 1 << pargs[1];
        const int offset_c = 1 << c_index;
        const int mask1 = offset1 - 1;
        const int mask2 = offset2 - 1;
        const int mask_c = offset_c - 1;

        int gw = 0, _0 = 0;

        if (c_index < pargs[0]){
            gw = label >> c_index << (c_index + 1);
            _0 = offset_c + (gw >> pargs[0] << (pargs[0] + 1)) + (gw & (offset1 - offset_c)) + (label & mask_c);
            _0 = offset_2 + (_0 >> pargs[1] << (pargs[1] + 1)) + (_0 & mask2)
        }elif(c_index < pargs[1]){
            gw = label >> pargs[0] << (pargs[0] + 1);
            _0 = offset_c + (gw >> c_index << (c_index + 1)) + (gw & (offset_c - offset1)) + (label & mask1);
            _0 = offset_2 + (_0 >> pargs[1] << (pargs[1] + 1)) + (_0 & mask2)
        }else{
            gw = label >> pargs[0] << (pargs[0] + 1);
            _0 = offset_2 + (gw >> pargs[1] << (pargs[1] + 1)) + (gw & (offset2 - offset1)) + (label & mask1);
            _0 = offset_c + (_0 >> c_index << (c_index + 1)) + (_0 & mask_c)
        }

        int _1 = _0 + offset1;

        complex<float> temp_0 = vec[_0];
        vec[_0] = vec[_1];
        vec[_1] = temp_0;
    }
    ''', 'PartI8x8Swap')


PartialIdentity_swap_more_double = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void PartI8x8Swap(int* pargs, int c_index, const complex<double>* mat, complex<double>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        const int offset1 = 1 << pargs[0];
        const int offset2 = 1 << pargs[1];
        const int offset_c = 1 << c_index;
        const int mask1 = offset1 - 1;
        const int mask2 = offset2 - 1;
        const int mask_c = offset_c - 1;

        int gw = 0, _0 = 0;

        if (c_index < pargs[0]){
            gw = label >> c_index << (c_index + 1);
            _0 = offset_c + (gw >> pargs[0] << (pargs[0] + 1)) + (gw & (offset1 - offset_c)) + (label & mask_c);
            _0 = offset_2 + (_0 >> pargs[1] << (pargs[1] + 1)) + (_0 & mask2)
        }elif(c_index < pargs[1]){
            gw = label >> pargs[0] << (pargs[0] + 1);
            _0 = offset_c + (gw >> c_index << (c_index + 1)) + (gw & (offset_c - offset1)) + (label & mask1);
            _0 = offset_2 + (_0 >> pargs[1] << (pargs[1] + 1)) + (_0 & mask2)
        }else{
            gw = label >> pargs[0] << (pargs[0] + 1);
            _0 = offset_2 + (gw >> pargs[1] << (pargs[1] + 1)) + (gw & (offset2 - offset1)) + (label & mask1);
            _0 = offset_c + (_0 >> c_index << (c_index + 1)) + (_0 & mask_c)
        }

        int _1 = _0 + offset1;

        complex<double> temp_0 = vec[_0];
        vec[_0] = vec[_1];
        vec[_1] = temp_0;
    }
    ''', 'PartI8x8Swap')


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
        Diagonal_Multiply_targ_single(
            (block_num,),
            (thread_per_block,),
            (t_index, mat, vec)
        )
    else:
        Diagonal_Multiply_targ_double(
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
        t_indexes[0], t_indexes[1] = t_indexes[1], t_indexes[0]

    gpu_indexes = cp.array(t_indexes, dtype=np.int32)

    if vec.dtype == np.complex64:
        Diagonal_Multiply_targs_single(
            (block_num,),
            (thread_per_block,),
            (gpu_indexes, mat, vec)
        )
    else:
        Diagonal_Multiply_targs_double(
            (block_num,),
            (thread_per_block,),
            (gpu_indexes, mat, vec)
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
        Based_InnerProduct_targ_single(
            (block_num,),
            (thread_per_block,),
            (t_index, mat, vec)
        )
    else:
        Based_InnerProduct_targ_double(
            (block_num,),
            (thread_per_block,),
            (t_index, mat, vec)
        )

    if sync:
        cp.cuda.Device().synchronize()


def Based_InnerProduct_targs(c_index, t_index, mat, vec, vec_bit, sync: bool = False):
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

    if c_index > t_index:
        indexes = cp.array([t_index, c_index], dtype=np.int32)
    else:
        indexes = cp.array([c_index, t_index], dtype=np.int32)

    if vec.dtype == np.complex64:
        Based_InnerProduct_targs_single(
            (block_num,),
            (thread_per_block,),
            (indexes, mat, vec)
        )
    else:
        Based_InnerProduct_targs_double(
            (block_num,),
            (thread_per_block,),
            (indexes, mat, vec)
        )

    if sync:
        cp.cuda.Device().synchronize()


def PartialIdentity_Multiply_targ(t_index, mat, vec, vec_bit, sync: bool = False):
    """
    partial identity diagonal matrix (2x2) dot vector
        [[1, 0],    *   vec
         [0, a]]
    """
    task_number = 1 << (vec_bit - 1)
    thread_per_block = min(256, task_number)
    block_num = task_number // thread_per_block

    if vec.dtype == np.complex64:
        PartialIdentity_Multiply_targ_single(
            (block_num,),
            (thread_per_block,),
            (t_index, mat, vec)
        )
    else:
        PartialIdentity_Multiply_targ_double(
            (block_num,),
            (thread_per_block,),
            (t_index, mat, vec)
        )

    if sync:
        cp.cuda.Device().synchronize()


def PartialIdentity_Multiply_ctargs(c_index, t_index, mat, vec, vec_bit, bit_pos, sync: bool = False):
    """
    partial identity diagonal matrix (4x4) dot vector
     e.g.   [[1, 0, 0, 0],    *   vec
             [0, 1, 0, 0],
             [0, 0, a, 0],
             [0, 0, 0, b]]
    """
    task_number = 1 << (vec_bit - 2)
    thread_per_block = min(256, task_number)
    block_num = task_number // thread_per_block

    if vec.dtype == np.complex64:
        PartialIdentity_Multiply_ctargs_single(
            (block_num,),
            (thread_per_block,),
            (mat, vec, c_index, t_index, bit_pos)
        )
    else:
        PartialIdentity_Multiply_ctargs_double(
            (block_num,),
            (thread_per_block,),
            (mat, vec, c_index, t_index, bit_pos)
        )

    if sync:
        cp.cuda.Device().synchronize()


def PartialIdentity_InnerProduct_ctargs(c_index, t_index, mat, vec, vec_bit, sync: bool = False):
    """
    partial identity based matrix (4x4) dot vector
     e.g.   [[1, 0, 0, 0],    *   vec
             [0, 1, 0, 0],
             [0, 0, a, b],
             [0, 0, c, d]]
    """
    task_number = 1 << (vec_bit - 2)
    thread_per_block = min(256, task_number)
    block_num = task_number // thread_per_block

    if vec.dtype == np.complex64:
        PartialIdentity_InnerProduct_ctargs_single(
            (block_num,),
            (thread_per_block,),
            (mat, vec, c_index, t_index)
        )
    else:
        PartialIdentity_InnerProduct_ctargs_double(
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
        t_indexes[0], t_indexes[1] = t_indexes[1], t_indexes[0]

    gpu_indexes = cp.array(t_indexes, dtype=np.int32)

    if vec.dtype == np.complex64:
        Completed_MxIP_targs_single(
            (block_num,),
            (thread_per_block,),
            (gpu_indexes, mat, vec)
        )
    else:
        Completed_MxIP_targs_double(
            (block_num,),
            (thread_per_block,),
            (gpu_indexes, mat, vec)
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
        t_indexes[0], t_indexes[1] = t_indexes[1], t_indexes[0]

    gpu_indexes = cp.array(t_indexes, dtype=np.int32)

    if vec.dtype == np.complex64:
        Completed_IPxIP_targs_single(
            (block_num,),
            (thread_per_block,),
            (gpu_indexes, mat, vec)
        )
    else:
        Completed_IPxIP_targs_double(
            (block_num,),
            (thread_per_block,),
            (gpu_indexes, mat, vec)
        )

    if sync:
        cp.cuda.Device().synchronize()


def RDiagonal_swap_targ(t_index, vec, vec_bit, sync: bool = False):
    """
    reverse diagonal matrix (2x2) dot vector
        [[0, 1],        *       vec
         [1, 0]]
    """
    task_number = 1 << (vec_bit - 1)
    thread_per_block = min(256, task_number)
    block_num = task_number // thread_per_block

    if vec.dtype == np.complex64:
        RDiagonal_swap_targ_single(
            (block_num,),
            (thread_per_block,),
            (t_index, vec)
        )
    else:
        RDiagonal_swap_targ_double(
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
        RDiagonal_MultiplySwap_targ_single(
            (block_num,),
            (thread_per_block,),
            (t_index, mat, vec)
        )
    else:
        RDiagonal_MultiplySwap_targ_double(
            (block_num,),
            (thread_per_block,),
            (t_index, mat, vec)
        )

    if sync:
        cp.cuda.Device().synchronize()


def PartialIdentity_MultiplySwap_ctargs(c_index, t_index, mat, vec, vec_bit, sync: bool = False):
    """
    partial identity rdiagonal matrix (4x4) dot vector
     e.g.   [[1, 0, 0, 0],    *   vec
             [0, 1, 0, 0],
             [0, 0, 0, a],
             [0, 0, b, 0]]
    """
    task_number = 1 << (vec_bit - 2)
    thread_per_block = min(256, task_number)
    block_num = task_number // thread_per_block

    if vec.dtype == np.complex64:
        PartialIdentity_MultiplySwap_ctargs_single(
            (block_num,),
            (thread_per_block,),
            (mat, vec, c_index, t_index)
        )
    else:
        PartialIdentity_MultiplySwap_ctargs_double(
            (block_num,),
            (thread_per_block,),
            (mat, vec, c_index, t_index)
        )

    if sync:
        cp.cuda.Device().synchronize()


def PartialIdentity_swap_targs(t_indexes, mat, vec, vec_bit, sync: bool = False):
    """
    partial identity rdiagonal matrix (4x4) dot vector
            [[1, 0, 0, 0],    *   vec
             [0, 0, 1, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1]]
    """
    task_number = 1 << (vec_bit - 2)
    thread_per_block = min(256, task_number)
    block_num = task_number // thread_per_block

    if t_indexes[0] > t_indexes[1]:
        t_indexes[0], t_indexes[1] = t_indexes[1], t_indexes[0]

    gpu_indexes = cp.array(t_indexes, dtype=np.int32)

    if vec.dtype == np.complex64:
        PartialIdentity_swap_targs_single(
            (block_num,),
            (thread_per_block,),
            (gpu_indexes, mat, vec)
        )
    else:
        PartialIdentity_swap_targs_double(
            (block_num,),
            (thread_per_block,),
            (gpu_indexes, mat, vec)
        )

    if sync:
        cp.cuda.Device().synchronize()


def PartialIdentity_swap_more(t_indexes, c_index, mat, vec, vec_bit, sync: bool = False):
    """
    partial identity rdiagonal matrix (8x8) dot vector
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

    if t_indexes[0] > t_indexes[1]:
        t_indexes[0], t_indexes[1] = t_indexes[1], t_indexes[0]

    gpu_indexes = cp.array(t_indexes, dtype=np.int32)

    if vec.dtype == np.complex64:
        PartialIdentity_swap_more_single(
            (block_num,),
            (thread_per_block,),
            (gpu_indexes, c_index, mat, vec)
        )
    else:
        PartialIdentity_swap_more_double(
            (block_num,),
            (thread_per_block,),
            (gpu_indexes, c_index, mat, vec)
        )

    if sync:
        cp.cuda.Device().synchronize()
