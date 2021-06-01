#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/5/28 6:08 下午
# @Author  : Han Yu
# @File    : refine_statevector_simulator

import numpy as np
import numba
from .._simulation import BasicSimulator

from QuICT.ops.linalg.gpu_calculator_refine import matrix_dot_vector_cuda

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
        vector = np.zeros(1 << qubit, dtype=np.complex64)
        vector[0] = 1
        vec = numba.cuda.to_device(vector)
        with numba.cuda.gpus[0]:
            for gate in small_gates:
                vec = matrix_dot_vector_cuda(
                    gate.compute_matrix,
                    gate.target + gate.controls,
                    vec,
                    qubit,
                    np.array(gate.affectArgs, dtype=np.int32)
                )
            vec.copy_to_host(vector)
        return vector
