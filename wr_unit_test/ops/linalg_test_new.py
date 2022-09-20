import time
import numpy as np
import cupy as cp 
import QuICT.ops.linalg.cpu_calculator_JIT as CPUCalculatorJIT
import QuICT.ops.linalg.cpu_calculator as CPUCalculator
import QuICT.ops.linalg.gpu_calculator as GPUCalculator
from QuICT.core.gate import *


def test_matrix_tensorI_speed():
    n, m = 2, 3
    I_N = np.identity(n)
    I_M = np.identity(m)
    qubit_num = 2  
    A = np.random.random((1 << qubit_num,1 << qubit_num )).astype(np.complex64)

#     start = time.time()
#     np.kron(np.kron(I_N, A), I_M)
#     print(f"matrix_tensorI_np time:{time.time() - start}")

    start = time.time()
    CPUCalculatorJIT.MatrixTensorI(A,n,m)
    print(f"matrix_tensorI_cpu time:{time.time() - start}")

#     start = time.time()
#     GPUCalculator.MatrixTensorI(A,n,m, gpu_out=False)
#     print(f"matrix_tensorI_gpu time:{time.time() - start}\n")
test_matrix_tensorI_speed()

def test_matrix_tensorI_speed():
    n, m = 2, 3
    I_N = np.identity(n)
    I_M = np.identity(m)
    qubit_num = 12
    A = np.random.random((1 << qubit_num,1 << qubit_num )).astype(np.complex64)

#     start = time.time()
#     np.kron(np.kron(I_N, A), I_M)
#     print(f"matrix_tensorI_np time:{time.time() - start}")

    start = time.time()
    CPUCalculatorJIT.MatrixTensorI(A,n,m)
    print(f"matrix_tensorI_cpu time:{time.time() - start}")

    start = time.time()
#     GPUCalculator.MatrixTensorI(A,n,m, gpu_out=False)
#     print(f"matrix_tensorI_gpu time:{time.time() - start}\n")
test_matrix_tensorI_speed()
