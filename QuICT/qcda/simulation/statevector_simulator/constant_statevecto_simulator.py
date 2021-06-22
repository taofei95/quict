#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/6/1 下午4:03
# @Author  : Han Yu
# @File    : constant_statevecto_simulator

import numpy as np
import cupy as cp
from typing import Union

from QuICT.ops.linalg.constant_cal_predata import gate_kernel_by_precision
from QuICT.qcda.simulation.utils.gate_based import GateMatrixs
from QuICT.core import *


class ConstantStateVectorSimulator:
    def __init__(self, circuit: Circuit, precision, device: int = 0):
        self.qubits = circuit.circuit_width()
        self.precision = precision
        self.gates = circuit.gates
        self.device = device

        # Initial gate_based algorithm
        self.gate_algorithm = gate_kernel_by_precision(self.precision)

    def __call__(self):
        # Special Case for no gate circuit
        if len(self.gates) == 0:
            vec = np.zeros(1 << self.qubits, dtype=self.precision)
            vec[0] = self.precision(1)
            return vec

        # Initial qubit's states
        self.vector = cp.empty(1 << self.qubits, dtype=self.precision)
        self.vector.put(0, self.precision(1))

        # Prepare gate matrixs
        self.gateM_manager = GateMatrixs(self.precision)
        for gate in self.gates:
            self.gateM_manager.build(gate)

        self.gateM_manager.concentrate_gate_matrixs()

        return self.run()

    def run(self) -> np.ndarray:
        """
        Get the state vector of circuit

        Args:
            circuit (Circuit): Input circuit to be simulated.

        Returns:
            np.ndarray: The state vector of input circuit.
        """
        with cp.cuda.Device(self.device):
            for gate in self.gates:
                matrix = self.gateM_manager.target_matrix(gate)

                self.gate_algorithm(
                    gate,
                    matrix,
                    self.vector,
                    self.qubits
                )

            return self.vector
