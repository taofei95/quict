import time
import numpy as np
import cupy as cp 
import QuICT.ops.linalg.cpu_calculator_JIT as CPUCalculatorJIT
import QuICT.ops.linalg.cpu_calculator as CPUCalculator
import QuICT.ops.linalg.gpu_calculator as GPUCalculator
from QuICT.core.gate import *

# def test_dot_speed():
#     qubit_num = 2  
#     A = np.random.random((1 << qubit_num,1 << qubit_num )).astype(np.complex64)
#     B = np.random.random((1 << qubit_num,1 << qubit_num )).astype(np.complex64)
#     # gpu_a = cp.array(A) 
#     # gpu_b = cp.array(B) 

#     # start = time.time()
#     # a = np.dot(A, B)
#     # print(f"dot_np time:{time.time() - start}")

#     start = time.time()
#     b = CPUCalculator.dot(A, B)
#     print(f"dot_cpu time:{time.time() - start}")

#     # start = time.time()
#     # c = GPUCalculator.dot(gpu_a, gpu_b, gpu_out=False)
#     # print(f"dot_gpu time:{time.time() - start}\n")
# test_dot_speed()

# def test_dot_speed():
#     qubit_num = 13
#     A = np.random.random((1 << qubit_num,1 << qubit_num )).astype(np.complex64)
#     B = np.random.random((1 << qubit_num,1 << qubit_num )).astype(np.complex64)
#     # gpu_a = cp.array(A) 
#     # gpu_b = cp.array(B) 

#     # start = time.time()
#     # a = np.dot(A, B)
#     # print(f"dot_np time:{time.time() - start}")

#     start = time.time()
#     b = CPUCalculator.dot(A, B)
#     print(f"dot_cpu time:{time.time() - start}")

#     # start = time.time()
#     # c = GPUCalculator.dot(gpu_a, gpu_b, gpu_out=False)
#     # print(f"dot_gpu time:{time.time() - start}\n")
# test_dot_speed()

# def test_tensor_speed():
#     qubit_num = 2
#     A = np.random.random((1 << qubit_num,1 << qubit_num )).astype(np.complex128)
#     B = np.random.random((1 << qubit_num,1 << qubit_num )).astype(np.complex128)
#     start = time.time()
#     np.kron(A, B)
#     print(f"tensor_np time:{time.time() - start}")

    
#     start = time.time()
#     CPUCalculator.tensor(A, B)
#     print(f"tensor_cpu time:{time.time() - start}")

#     start = time.time()
#     GPUCalculator.tensor(A, B, gpu_out=False)
#     print(f"tensor_gpu time:{time.time() - start}\n")
# test_tensor_speed()

# def test_tensor_speed():
#     qubit_num = 4
#     A = np.random.random((1 << qubit_num,1 << qubit_num )).astype(np.complex128)
#     B = np.random.random((1 << qubit_num,1 << qubit_num )).astype(np.complex128)
#     start = time.time()
#     np.kron(A, B)
#     print(f"tensor_np time:{time.time() - start}")

    
#     start = time.time()
#     CPUCalculator.tensor(A, B)
#     print(f"tensor_cpu time:{time.time() - start}")

#     start = time.time()
#     GPUCalculator.tensor(A, B, gpu_out=False)
#     print(f"tensor_gpu time:{time.time() - start}\n")
# test_tensor_speed()

# def test_matrix_tensorI_speed():
#     n, m = 2, 3
#     I_N = np.identity(n)
#     I_M = np.identity(m)
#     qubit_num = 2  
#     A = np.random.random((1 << qubit_num,1 << qubit_num )).astype(np.complex128)

# #     start = time.time()
# #     np.kron(np.kron(I_N, A), I_M)
# #     print(f"matrix_tensorI_np time:{time.time() - start}")

#     start = time.time()
#     CPUCalculator.MatrixTensorI(A,n,m)
#     print(f"matrix_tensorI_cpu time:{time.time() - start}")

# #     start = time.time()
# #     GPUCalculator.MatrixTensorI(A,n,m, gpu_out=False)
# #     print(f"matrix_tensorI_gpu time:{time.time() - start}\n")
# test_matrix_tensorI_speed()

# def test_matrix_tensorI_speed():
#     n, m = 2, 3
#     I_N = np.identity(n)
#     I_M = np.identity(m)
#     qubit_num = 12
#     A = np.random.random((1 << qubit_num,1 << qubit_num )).astype(np.complex128)

# #     start = time.time()
# #     np.kron(np.kron(I_N, A), I_M)
# #     print(f"matrix_tensorI_np time:{time.time() - start}")

#     start = time.time()
#     CPUCalculator.MatrixTensorI(A,n,m)
#     print(f"matrix_tensorI_cpu time:{time.time() - start}")

#     start = time.time()
# #     GPUCalculator.MatrixTensorI(A,n,m, gpu_out=False)
# #     print(f"matrix_tensorI_gpu time:{time.time() - start}\n")
# test_matrix_tensorI_speed()

# def test_multiply_speed():
#     qubit_num = 2  
#     A = np.random.random((1 << qubit_num,1 << qubit_num )).astype(np.complex64)
#     B = np.random.random((1 << qubit_num,1 << qubit_num )).astype(np.complex64)

#     start = time.time()
#     np.multiply(A, B)
#     print(f"multiply_np time:{time.time() - start}")
 
#     start = time.time()
#     CPUCalculator.multiply(A, B)
#     print(f"multiply_cpu time:{time.time() - start}\n")
# test_multiply_speed()

# def test_multiply_speed():
#     qubit_num = 5  
#     A = np.random.random((1 << qubit_num,1 << qubit_num )).astype(np.complex64)
#     B = np.random.random((1 << qubit_num,1 << qubit_num )).astype(np.complex64)
    
#     start = time.time()
#     np.multiply(A, B)
#     print(f"multiply_np time:{time.time() - start}")
 
#     start = time.time()
#     CPUCalculator.multiply(A, B)
#     print(f"multiply_cpu time:{time.time() - start}\n")
# test_multiply_speed()
 
# def test_vectorpermutation_speed():
#     mapping = list(range(qubit_num * 2))[::-1]
#     mapping = np.array(mapping)
#     np.random.shuffle(mapping)
#     start = time.time()
#     CPUCalculator.VectorPermutation(vector, mapping)
#     print(f"vectorpermutation_cpu time:{time.time() - start}")

#     start = time.time()
#     GPUCalculator.VectorPermutation(C, mapping, changeInput=False, gpu_out=True)
#     print(f"vectorpermutation_gpu time:{time.time() - start}\n")
# test_vectorpermutation_speed()

# def test_matrixpermutation_speed():
#     mapping = list(range(qubit_num))[::-1]
#     mapping = np.array(mapping)
#     np.random.shuffle(mapping)
#     start = time.time()
#     CPUCalculator.MatrixPermutation(A, mapping)
#     print(f"mmatrixpermutation_cpu time:{time.time() - start}")

#     start = time.time()
#     GPUCalculator.MatrixPermutation(A, mapping, changeInput=False, gpu_out=True)
#     print(f"mmatrixpermutation_gpu time:{time.time() - start}\n")
# test_matrixpermutation_speed()

# def test_matrix_dot_vector_speed():
#     start = time.time()
#     CPUCalculator.tensor(A, B)
#     print(f"matrix_dot_vector_cpu time:{time.time() - start}")

#     start = time.time()
#     GPUCalculator.tensor(A, B)
#     print(f"matrix_dot_vector_gpu time:{time.time() - start}")
# test_matrix_dot_vector_speed()

# def test_matrixpermutation_speed():
#     qubit_num = 2
#     A = np.random.random((1 << qubit_num,1 << qubit_num )).astype(np.complex128)
#     gpu_a = cp.array(A) 

#     mapping = list(range(qubit_num))[::-1]
#     mapping = np.array(mapping)
#     np.random.shuffle(mapping)

#     start = time.time()
#     CPUCalculator.MatrixPermutation(A, mapping)
#     print(f"mmatrixpermutation_cpu time:{time.time() - start}")

#     start = time.time()
#     GPUCalculator.MatrixPermutation(gpu_a, mapping, changeInput=False, gpu_out=True)
#     print(f"mmatrixpermutation_gpu time:{time.time() - start}\n")
# test_matrixpermutation_speed()

# def test_matrixpermutation_speed():
#     qubit_num = 15
#     A = np.random.random((1 << qubit_num,1 << qubit_num )).astype(np.complex128)
#     gpu_a = cp.array(A) 

#     mapping = list(range(qubit_num))[::-1]
#     mapping = np.array(mapping)
#     np.random.shuffle(mapping)
    
#     start = time.time()
#     CPUCalculator.MatrixPermutation(A, mapping)
#     print(f"mmatrixpermutation_cpu time:{time.time() - start}")

#     start = time.time()
#     GPUCalculator.MatrixPermutation(gpu_a, mapping, changeInput=False, gpu_out=True)
#     print(f"mmatrixpermutation_gpu time:{time.time() - start}\n")
# test_matrixpermutation_speed()

# def test_matrix_dot_vector_speed():
#     qubit_num = 2  
#     A = np.random.random((1 << qubit_num,1 << qubit_num )).astype(np.complex64)
#     B = np.random.random((1 << qubit_num,1 << qubit_num )).astype(np.complex64)
#     gpu_a = cp.array(A) 
#     gpu_b = cp.array(B) 

#     start = time.time()
#     CPUCalculator.tensor(A, B)
#     print(f"matrix_dot_vector_cpu time:{time.time() - start}")

#     start = time.time()
#     GPUCalculator.tensor(gpu_a, gpu_b)
#     print(f"matrix_dot_vector_gpu time:{time.time() - start}")
# test_matrix_dot_vector_speed()

# def test_matrix_dot_vector_speed():
#     qubit_num = 2  
#     A = np.random.random((1 << qubit_num,1 << qubit_num )).astype(np.complex64)
#     B = np.random.random((1 << qubit_num,1 << qubit_num )).astype(np.complex64)
#     gpu_a = cp.array(A) 
#     gpu_b = cp.array(B) 

#     start = time.time()
#     CPUCalculator.tensor(A, B)
#     print(f"matrix_dot_vector_cpu time:{time.time() - start}")

#     start = time.time()
#     GPUCalculator.tensor(gpu_a, gpu_b)
#     print(f"matrix_dot_vector_gpu time:{time.time() - start}")
# test_matrix_dot_vector_speed()



# def test_dot_speed():
#     qubit_num = 2  
#     A = np.random.random((1 << qubit_num,1 << qubit_num )).astype(np.complex64)
#     B = np.random.random((1 << qubit_num,1 << qubit_num )).astype(np.complex64)
   

#     start = time.time()
#     b = CPUCalculatorJIT.dot(A, B)
#     print(f"dot_cpu_jit time:{time.time() - start}")

# test_dot_speed()

# def test_dot_speed():
#     qubit_num = 9
#     A = np.random.random((1 << qubit_num,1 << qubit_num )).astype(np.complex64)
#     B = np.random.random((1 << qubit_num,1 << qubit_num )).astype(np.complex64)
   
#     start = time.time()
#     b = CPUCalculatorJIT.dot(A, B)
#     print(f"dot_cpu_jit time:{time.time() - start}")
# test_dot_speed()

