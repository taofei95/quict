#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/5/28 6:08 下午
# @Author  : Han Yu
# @File    : refine_statevector_simulator

import time

import numpy as np
import numba
from .._simulation import BasicSimulator

from QuICT.ops.linalg.gpu_calculator_refine import matrix_dot_vector_cuda
from QuICT.core import *

class RefineStateVectorSimulator(BasicSimulator):
    @classmethod
    def run(cls, circuit: Circuit) -> np.ndarray:
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
        small_gates = BasicSimulator.pretreatment(circuit)
        # vector = np.zeros(1 << qubit, dtype=np.complex64)
        # vector[0] = 1
        with numba.cuda.gpus[0]:
            # anc = numba.cuda.to_device(np.zeros(1 << qubit, dtype=np.complex64))
            fy_start = time.time()
            anc = numba.cuda.device_array((1 << qubit, ), dtype=np.complex64)
            vec = numba.cuda.device_array((1 << qubit, ), dtype=np.complex64)
            vec[0] = 1
            fy_end = time.time()
            time_start = time.time()
            for gate in small_gates:
                matrix_dot_vector_cuda(
                    gate.compute_matrix,
                    gate.targets + gate.controls,
                    vec,
                    qubit,
                    np.array(gate.affectArgs, dtype=np.int32),
                    auxiliary_vec = anc
                )
                anc, vec = vec, anc
            time_end = time.time()
            print("cal time", time_end - time_start)
            print("malloc time", fy_end - fy_start)
            return vec
        # return vector

