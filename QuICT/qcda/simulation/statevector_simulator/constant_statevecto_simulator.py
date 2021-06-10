#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/6/1 下午4:03
# @Author  : Han Yu
# @File    : constant_statevecto_simulator

import time

import numpy as np
import numba

from QuICT.ops.linalg.gpu_constant_calculator_refine import gate_dot_vector_cuda
from QuICT.ops.linalg.constant_cal_predata import gate_dot_vector_predata
from QuICT.qcda.simulation.utils.gate_based import GateMatrixs
from QuICT.core import *


class ConstantStateVectorSimulator:
    @staticmethod
    def run(circuit: Circuit) -> np.ndarray:
        """
        Get the state vector of circuit

        Args:
            circuit (Circuit): Input circuit to be simulated.

        Returns:
            np.ndarray: The state vector of input circuit.
        """

        qubit = circuit.circuit_width()
        if len(circuit.gates) == 0:
            vector = np.zeros(1 << qubit, dtype=np.complex64)
            vector[0] = 1 + 0j
            return vector
        # small_gates = BasicSimulator.pretreatment(circuit)
        gates = circuit.gates
        # vector = np.zeros(1 << qubit, dtype=np.complex64)
        # vector[0] = 1

        with numba.cuda.gpus[0]:
            # anc = numba.cuda.to_device(np.zeros(1 << qubit, dtype=np.complex64))
            fy_start = time.time()
            vec = numba.cuda.device_array((1 << qubit, ), dtype=np.complex64)
            vec[0] = 1
            fy_end = time.time()
            time_start = time.time()
            for gate in gates:
                gate_dot_vector_cuda(
                    gate,
                    vec,
                    qubit
                )
            time_end = time.time()
            print("cal time", time_end - time_start)
            print("malloc time", fy_end - fy_start)
            # return vec.copy_to_host()
            return vec
        # return vector

    @staticmethod
    def run_predata_ot(circuit: Circuit) -> np.ndarray:
        """
        Get the state vector of circuit

        Args:
            circuit (Circuit): Input circuit to be simulated.

        Returns:
            np.ndarray: The state vector of input circuit.
        """

        qubit = circuit.circuit_width()
        if len(circuit.gates) == 0:
            vector = np.zeros(1 << qubit, dtype=np.complex64)
            vector[0] = 1 + 0j
            return vector

        gates = circuit.gates

        with numba.cuda.gpus[0]:
            fy_start = time.time()
            gatematrixG = GateMatrixs()
            for gate in gates:
                gatematrixG.build(gate)
            print(f"Data gather time: {time.time() - fy_start}")

            start_time = time.time()
            gatematrixG.concentrate_gate_matrixs()
            vec = numba.cuda.device_array((1 << qubit, ), dtype=np.complex64)
            vec[0] = 1
            print(f"Data transfer time: {time.time() - start_time}")
            fy_end = time.time()
            time_start = time.time()
            for gate in gates:
                matrix = gatematrixG.target_matrix(gate)

                gate_dot_vector_predata(
                    gate,
                    matrix,
                    vec,
                    qubit
                )
            time_end = time.time()
            print("cal time", time_end - time_start)
            print("malloc time", fy_end - fy_start)

            return vec
