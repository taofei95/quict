import numpy as np

from random import randint
from time import time
from typing import *

from numba import cuda as nb_cuda


def test_perf_copy_diff():
    with nb_cuda.gpus[-1]:
        qubit_num = 28
        gate_cnt = 2000
        state_vector_transfer_time = 0.0
        gate_mat_transfer_time = 0.0
        gate_arg_transfer_time = 0.0
        data_gather_time = 0.0
        compact_mat_transfer_time = 0.0
        compact_arg_transfer_time = 0.0
        compact_gates: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
        mat_sz_sum = 0
        mat_arg_num_sum = 0

        vec = np.random.rand(1 << qubit_num) + np.random.rand(1 << qubit_num) * 1.0j
        start_time = time()
        nb_cuda.to_device(vec)
        end_time = time()
        state_vector_transfer_time += (end_time - start_time) * 1000

        for _ in range(gate_cnt):
            affect_num = randint(1, 3)
            mat_sz_sum += (1 << (affect_num << 1))
            mat_arg_num_sum += affect_num
            mat = np.random.rand(1 << affect_num, 1 << affect_num) + \
                  np.random.rand(1 << affect_num, 1 << affect_num) * 1.0j
            arg = np.random.rand(affect_num)
            arg_sorted = np.sort(arg)
            compact_gates.append((affect_num, mat, arg, arg_sorted))

            start_time = time()
            nb_cuda.to_device(mat)
            end_time = time()
            gate_mat_transfer_time += (end_time - start_time) * 1000

            start_time = time()
            nb_cuda.to_device(arg)
            nb_cuda.to_device(arg_sorted)
            end_time = time()
            gate_arg_transfer_time += (end_time - start_time) * 1000

        start_time = time()
        compact_gate_mat = np.empty(shape=mat_sz_sum, dtype=np.complex64)
        compact_gate_arg = np.empty(shape=mat_arg_num_sum, dtype=np.int)
        compact_gate_arg_sorted = np.empty(shape=mat_arg_num_sum, dtype=np.int)
        arg_strides = np.empty(shape=gate_cnt, dtype=np.int)
        mat_strides = np.empty(shape=gate_cnt, dtype=np.int)
        arg_offset = 0
        mat_offset = 0

        for i in range(gate_cnt):
            affect_num, mat, arg, arg_sorted = compact_gates[i]
            mat_strides[i] = mat_offset
            arg_strides[i] = arg_offset
            compact_gate_mat[mat_offset:mat_offset + (1 << (affect_num << 1))] = mat.flatten()[:]
            compact_gate_arg[arg_offset:arg_offset + affect_num] = arg[:]
            compact_gate_arg_sorted[arg_offset:arg_offset + affect_num] = arg_sorted[:]
            mat_offset += (1 << (affect_num << 1))
            arg_offset += affect_num

        end_time = time()
        data_gather_time += (end_time - start_time) * 1000

        start_time = time()
        nb_cuda.to_device(compact_gate_mat)
        nb_cuda.to_device(mat_strides)
        end_time = time()
        compact_mat_transfer_time += (end_time - start_time) * 1000

        start_time = time()
        nb_cuda.to_device(compact_gate_arg)
        nb_cuda.to_device(compact_gate_arg_sorted)
        nb_cuda.to_device(arg_strides)
        end_time = time()
        compact_arg_transfer_time += (end_time - start_time) * 1000

        print()
        print(f"Qubit number: {qubit_num}")
        print(f"Gate number: {gate_cnt}")
        print("Data transfer time:")
        print(f"State vector: {state_vector_transfer_time:0.4f} ms")
        print(f"Gate mat: {gate_mat_transfer_time:0.4f} ms")
        print(f"Gate args: {gate_arg_transfer_time:0.4f} ms")
        print(f"Gate gather time: {data_gather_time:0.4f} ms")
        print(f"Compact mat: {compact_mat_transfer_time:0.4f} ms")
        print(f"Compact args: {compact_arg_transfer_time:0.4f} ms")
