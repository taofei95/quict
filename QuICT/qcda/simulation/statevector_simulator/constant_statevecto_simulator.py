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
from QuICT.core import *


class CRZGateGeneator:
    def __init__(self):
        self.pre_build = {}

    def build(self, parg):
        if parg in self.pre_build.keys():
            return self.pre_build[parg]
        
        matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, np.exp(-parg / 2 * 1j), 0],
            [0, 0, 0, np.exp(parg / 2 * 1j)]
        ], dtype=np.complex64)

        self.pre_build[parg] = numba.cuda.to_device(matrix)

        return self.pre_build[parg]  

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
    def run_predata(circuit: Circuit) -> np.ndarray:
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
            HGate_matrix = np.array([
                [1 / np.sqrt(2), 1 / np.sqrt(2)],
                [1 / np.sqrt(2), -1 / np.sqrt(2)]
            ], dtype=np.complex64)
            HGate_matrix = numba.cuda.to_device(HGate_matrix)
            CRZ_generator = CRZGateGeneator()

            vec = numba.cuda.device_array((1 << qubit, ), dtype=np.complex64)
            vec[0] = 1
            fy_end = time.time()
            time_start = time.time()
            for gate in gates:
                gname = gate.name.split("_")[0]
                if gname == "CRzGate":
                    mat = CRZ_generator.build(gate.parg)
                else:
                    mat = HGate_matrix

                gate_dot_vector_predata(
                    gate,
                    mat,
                    vec,
                    qubit
                )
            time_end = time.time()
            print("cal time", time_end - time_start)
            print("malloc time", fy_end - fy_start)
            # return vec.copy_to_host()
            return vec
        # return vector
