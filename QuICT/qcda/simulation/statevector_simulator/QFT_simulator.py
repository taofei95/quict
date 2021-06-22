#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/5/28 6:08 下午
# @Author  : Han Yu
# @File    : refine_statevector_simulator

import numpy as np
from tempfile import mkdtemp
import os.path as path

from QuICT.core import *
from QuICT.qcda.simulation.utils.gate_kernel import *


class QFTSimulator:
    def __init__(self, circuit: Circuit):
        self.circuit = circuit
        self.qubits = self.circuit.circuit_width()
        self.vector = np.zeros(1 << self.qubits, dtype=np.complex64)
        self.vector[0] = 1 + 0j

    def _gate_matrix_dot_vector(self, gate):
        # gate switch
        gate_name = gate.name.split("_")[0]

        if gate_name == "HGate":
            Hgate_kernel(self.qubits - 1 - gate.targ, self.vector)
        elif gate_name == "CRzGate":
            c_index = self.qubits - 1 - gate.carg
            t_index = self.qubits - 1 - gate.targ
            if t_index > c_index:
                CRZgate_kernel_target(c_index, t_index, gate.compute_matrix, self.vector)
            else:
                CRZgate_kernel_control(c_index, t_index, gate.compute_matrix, self.vector)

    def run(self) -> np.ndarray:
        """
        Get the state vector of circuit

        Args:
            circuit (Circuit): Input circuit to be simulated.

        Returns:
            np.ndarray: The state vector of input circuit.
        """

        if len(self.circuit.gates) == 0:
            return self.vector

        gates = self.circuit.gates

        for gate in gates:
            self._gate_matrix_dot_vector(gate)
        
        return self.vector
