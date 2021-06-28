#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/6/1 下午4:03
# @Author  : Han Yu
# @File    : constant_statevecto_simulator

import numpy as np
import cupy as cp
import time

from QuICT.qcda.simulation.proxy_simulator.gate_func_single import \
    GateFuncS, GateFuncMS
from QuICT.qcda.simulation.utils.gate_based import GateMatrixs
from QuICT.core import *


class ConstantStateVectorSimulator:
    def __init__(self, circuit: Circuit, precision, device: int = 0):
        self._qubits = int(circuit.circuit_width())
        self._precision = precision
        self._gates = circuit.gates
        self._device = device

        # Initial gate_based algorithm
        # self._algorithm = gate_kernel_single if self._precision == np.complex64 else gate_kernel_double

    @property
    def qubits(self):
        return self._qubits

    @qubits.setter
    def qubits(self, qubit: int):
        self._qubits = qubit

    @property
    def precision(self):
        return self._precision

    @precision.setter
    def precision(self, precision):
        if precision != self._precision:
            self._precision = precision
            # self._algorithm = gate_kernel_single if self._precision == np.complex64 else gate_kernel_double

    @property
    def gates(self):
        return self._gates

    @property
    def algorithm(self):
        return self._algorithm

    @property
    def vector(self):
        return self._vector

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device: int):
        self._device = device

    def run(self) -> np.ndarray:
        """
        Get the state vector of circuit

        Args:
            circuit (Circuit): Input circuit to be simulated.

        Returns:
            np.ndarray: The state vector of input circuit.
        """
        self.initial_vector_state()

        with cp.cuda.Device(self._device):
            for gate in self._gates:
                self.exec(gate)
    
        return self._vector

    def initial_vector_state(self):
        vector_size = 1 << int(self._qubits)
        # Special Case for no gate circuit
        if len(self._gates) == 0:
            self._vector = np.zeros(vector_size, dtype=self._precision)
            self._vector[0] = self._precision(1)
            return

        # Initial qubit's states
        with cp.cuda.Device(self._device):
            self._vector = cp.empty(vector_size, dtype=self._precision)
            self._vector.put(0, self._precision(1))

        self._initial_gate_matrix_optimized()

    def _initial_gate_matrix_optimized(self):
        # Prepare gate matrixs optimizer
        self.gateM_optimizer = GateMatrixs(self._precision, self._device)
        for gate in self._gates:
            self.gateM_optimizer.build(gate)

        self.gateM_optimizer.concentrate_gate_matrixs()

    def exec(self, gate):
        matrix = self.gateM_optimizer.target_matrix(gate)

        print("here")

        if gate.type() == GATE_ID["H"]:
            print("H")

        elif gate.type() == GATE_ID["CRz"]:
            print("CRz")
            
        else:
            raise KeyError(f"Unsupported gate type {gate.type()}")

    def calculator(self, gate, _0_1, cindex):
        matrix = self.gateM_optimizer.target_matrix(gate)

        GateFuncMS.CRzGate_matrixdot_pc(
            _0_1,
            cindex,
            mat = matrix,
            vec = self._vector,
            vec_bit = self._qubits
        )

    def get_GateM(self, gate):
        return self.gateM_optimizer.target_matrix(gate)

    def reset_vector(self, new_vector, front: bool = True):
        new_vector_size = new_vector.size
        assert(new_vector_size == 1 << (int(self._qubits) - 1))

        if front:
            self._vector[:new_vector_size] = new_vector
        else:
            self._vector[new_vector_size:] = new_vector
