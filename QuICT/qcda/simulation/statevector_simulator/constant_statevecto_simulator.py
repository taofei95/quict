#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/6/1 下午4:03
# @Author  : Han Yu
# @File    : constant_statevecto_simulator

import numpy as np
import cupy as cp

from QuICT.qcda.simulation.proxy_simulator.gate_func_single import GateFuncS
from QuICT.qcda.simulation.proxy_simulator.gate_func_double import GateFuncD
from QuICT.qcda.simulation import BasicSimulator
from QuICT.core import *


class ConstantStateVectorSimulator(BasicSimulator):
    def __init__(self, circuit: Circuit, precision, device: int = 0, sync: bool = False):
        BasicSimulator.__init__(self, circuit, precision, device)
        self._sync = sync

        # Initial vector state
        self.initial_vector_state()

        # Initial gate_based algorithm
        self._algorithm = GateFuncS if self._precision == np.complex64 else GateFuncD

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

    def run(self) -> np.ndarray:
        """
        Get the state vector of circuit

        Args:
            circuit (Circuit): Input circuit to be simulated.

        Returns:
            np.ndarray: The state vector of input circuit.
        """
        with cp.cuda.Device(self._device):
            for gate in self._gates:
                self.exec(gate)
    
        return self.vector

    def exec(self, gate):
        matrix = self.get_Matrix(gate)

        if gate.type() == GATE_ID["H"]:
            t_index = self._qubits - 1 - gate.targ
            self._algorithm.HGate_matrixdot(
                t_index,
                matrix,
                self._vector,
                self._qubits,
                self._sync
            )
        elif gate.type() == GATE_ID["CRz"]:
            t_index = self._qubits - 1 - gate.targ
            c_index = self._qubits - 1 - gate.carg
            self._algorithm.CRzGate_matrixdot(
                c_index,
                t_index,
                matrix,
                self._vector,
                self._qubits,
                self._sync
            )
        else:
            raise KeyError(f"Unsupported gate type {gate.type()}")
