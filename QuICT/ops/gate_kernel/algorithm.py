import cupy as cp
import numpy as np

import random

"""
By matrix type: normal matrix and diagonal matrix (divided by gate.is_diagonal())
By qubit index: 1-qreg and 2-qregs (divided by gate.is_single())
Special gates: Measure, Perm, Unitary, Barrier (gate.is_special())

For normal matrix:
1-qreg: inner product
    [[c1, c2],    *   [v0,      = [c1*v0 + c2*v1,
     [c3, c4]]         v1]         c3*v0 + c4*v1]

    Assume 1-qreg in k index, v1 = v0 | 1 << k

    Suitable Gates: ['H', 'SX', 'SY', 'SW', 'U2', 'U3', 'Rx', 'Ry']

2-qregs or more: inner product
    [[c1,c2,c3,c4],     *   [v0,    = [c1*v0 + c2*v1 + c3*v2 + c4*v3,
     [c5,c6,c7,c8],          v1,       c5*v0 + c6*v1 + c7*v2 + c8*v3,
     [c9,c10,c11,c12],       v2,       c9*v0 + c10*v1 + c11*v2 + c12*v3,
     [c13,c14,c15,c16]]      v3]       c13*v0 + c14*v1 + c15*v2 + c16*v3]

    Let t and c represent the indexes of qregs, where t is larger than c. Also, assume
    offset1 = 1 << c, offset2 = 1 << t. Therefore,
        v1 = v0 | offset1
        v2 = v0 | offset2
        v3 = v0 | offset1 + offset2

    Suitable Gates: []

For diagonal matrix:
1-qreg: multiply
    [[c1, 0],   *   [v0,    =   [c1*v0,  
     [0, c2]]        v1]         c2*v1]

    Assume 1-qreg in k index, v1 = v0 | 1 << k

    Suitable Gates: ['S', 'S_dagger', 'Rz', 'Phase']

2-qregs or more: multiply
    [[c1, 0, 0, 0],     *   [v0,    = [c1*v0,
     [0, c2, 0, 0],          v1,       c2*v1,
     [0, 0, c3, 0],          v2,       c3*v2,
     [0, 0, 0, c4]]          v3]       c4*v3]

    Let t and c represent the indexes of qregs, where t is larger than c. Also, assume
    offset1 = 1 << c, offset2 = 1 << t. Therefore,
        v1 = v0 | offset1
        v2 = v0 | offset2
        v3 = v0 | offset1 + offset2

    Suitable Gates: ['CZ', 'RZZ', ]


For special matrix including reverse diagonal matrix, identity/half identity matrix, and etc:
reverse diagonal matrix: [[0, c1],
                          [c2, 0]]
    Gates: ['Y', 'X']
    
identity matrix: [[1, 0],
                  [0, 1]]
    Gates：['ID', ]
    
partial identity matrix: [[1, 0],          [[1, 0, 0, 0],      [[1, 0, 0, 0],   ...
                          [0, c1]]          [0, 1, 0, 0],       [0, 1, 0, 0],   
                                            [0, 0, c1, 0],      [0, 0, 1, 0],   
                                            [0, 0, 0, c2]]      [0, 0, 0, c1]]
    Gates: ['CXX', 'CU1', 'CRz', 'CZ', 'T_dagger', 'T', 'U1', 'Z', 'S_dagger']

    [[1, 0, 0, 0],
     [0, 1, 0, 0], 
     [0, 0, c1, c2],
     [0, 0, c3, c4]]
    Gates: ['CU3', 'CH', ]

    [[1, 0, 0, 0],
     [0, 1, 0, 0], 
     [0, 0, 0, c1],  
     [0, 0, c2, 0]]
    Gates: ['CY', 'CX', ]

    [[1, 0, 0, 0],
     [0, 0, c1, 0],      
     [0, c2, 0, 0],
     [0, 0, 0, 1]]
    Gates: [Swap, ]

Completed case:
    [[c0, 0, 0, 0],
     [0, c1, c2, 0],
     [0, c3, c4, 0],
     [0, 0, 0, c5]]
    Gates: ['FSim', ]

    [[c1, 0, 0, c2],
     [0, c3, c4, 0],
     [0, c5, c6, 0],
     [c7, 0, 0, c8]]
    Gates: ['RYY', 'RXX', ]


Measure Gate
reset Gate
permutate Gate
Barrier Gate
Unitary Gate
"""

"""
Algorithm:
Switch *cases* by gate type

Case: Normal Matrix (could combine two within one algorithm.)
    1-qreg: 
        Get *gate_matrix(2*2)*      -- GateMatrix
        Calculate index (v1) by given v0
        Do inner product
    2-qreg:
        Get *gate_matrix(4*4)*      -- GateMatrix
        Calculate index (v1, v2, v3) by given v0    
        Do inner product
Case: diagonal matrix (could combine two within one algorithm.)
    1-qreg: 
        Get *gate_matrix(2*2)*      -- GateMatrix
        Calculate index (v1) by given v0
        Do multiply
    2-qreg:
        Get *gate_matrix(4*4)*      -- GateMatrix
        Calculate index (v1, v2, v3) by given v0    
        Do multiply
Case: reverse diagonal matrix (may need divided by qreg number)
    Get *gate_matrix*
    Calculate indexes by given v0
    Swap
    Do multiply
Case: identity matrix:
    pass?
Case: partial identity matrix:
    Get available part (non-identity part)
    Depending on the type of available part, do correlated case.
Case: completed matrix:
    Divided into two parts
        e.g.    [[c1, 0, 0, c2],    *   [v0,   ==> [[c1, c2], * [v0, 
                 [0, c3, c4, 0],         v1,        [c7, c8]]    v3]
                 [0, c5, c6, 0],         v2,       [[c3, c4], * [v1,
                 [c7, 0, 0, c8]]         v3]        [c5, c6]]    v2]
    Do inner product or multiply two-by-two
"""

"""
TODO: 1. *CRz* & one of completed matrix (Finish)
      2. Comparsion speed with gate_based kernel function (Finish)
      3. Analysis Result (Finish)

TODO: 1. data_t support -> float/double (Finish)
      2. gate matrix permutation (followed by c-t or low-high) (...)
      3. apply based multiply and inner product (Finish)
      4. think new interface for customized vector state 
"""


# multiply_2args_single_kernel = cp.RawModule(code=r'''
#     #include <cupy/complex.cuh>
#     template<typename T>
#     __global__ void MultiplyTwoArgs(int* pargs, const complex<T>* mat, complex<T>* vec, bool reverse) {
#         int label = blockDim.x * blockIdx.x + threadIdx.x;

#         const int offset1 = 1 << pargs[0];
#         const int offset2 = 1 << pargs[1];
#         const int mask1 = offset1 - 1;
#         const int mask2 = offset2 - 1;

#         int gw = label >> pargs[0] << (pargs[0] + 1);
#         int _0 = (gw >> pargs[1] << (pargs[1] + 1)) + (gw & (offset2 - offset1)) + (label & mask1);

#         int _1=0, _2=0, _3=0;

#         if(reverse){
#             _1 = _0 + offset2;
#             _2 = _0 + offset1;
#             _3 = _2 + offset2;
#         }else{
#             _1 = _0 + offset1;
#             _2 = _0 + offset2;
#             _3 = _2 + offset1;
#         }


#         vec[_0] = vec[_0]*mat[0];
#         vec[_1] = vec[_1]*mat[5];
#         vec[_2] = vec[_2]*mat[10];
#         vec[_3] = vec[_3]*mat[15];
#     }
#     ''', options=('-std=c++11',), name_expressions=['MultiplyTwoArgs<float>', 'MultiplyTwoArgs<double>'])
