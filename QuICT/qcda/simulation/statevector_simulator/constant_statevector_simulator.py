#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/6/28 下午4:50
# @Author  : Kaiqi Li
# @File    : constant_statevecto_simulator

import numpy as np
import cupy as cp

from QuICT.ops.utils import LinAlgLoader
from QuICT.qcda.simulation import BasicSimulator
from QuICT.core import *


class ConstantStateVectorSimulator(BasicSimulator):
    """
    The simulator for qubits' vector state.

    Args:
        circuit (Circuit): The quantum circuit.
        precision [np.complex64, np.complex128]: The precision for the circuit and qubits.
        gpu_device_id (int): The GPU device ID.
        sync (bool): Sync mode or Async mode.
    """
    def __init__(self, circuit: Circuit, precision = np.complex64, gpu_device_id: int = 0, sync: bool = False):
        BasicSimulator.__init__(self, circuit, precision, gpu_device_id)
        self._sync = sync

        # Initial vector state
        self.initial_vector_state()

        # Initial simulator with limit_qubits
        self._algorithm = LinAlgLoader(device="GPU", enable_gate_kernel=True, enable_multigpu_gate_kernel=False)

    def initial_vector_state(self):
        """
        Initial qubits' vector states.
        """
        vector_size = 1 << int(self._qubits)
        # Special Case for no gate circuit
        if len(self._gates) == 0:
            self._vector = np.zeros(vector_size, dtype=self._precision)
            self._vector[0] = self._precision(1)
            return

        # Initial qubit's states
        with cp.cuda.Device(self._device_id):
            self._vector = cp.empty(vector_size, dtype=self._precision)
            self._vector.put(0, self._precision(1))

    def run(self) -> np.ndarray:
        """
        Get the state vector of circuit
        """
        with cp.cuda.Device(self._device_id):
            for gate in self._gates:
                self.exec(gate)
    
        return self.vector

    def exec(self, gate):
        if gate.type() == GATE_ID["H"]:
            matrix = self.get_Matrix(gate)
            t_index = self._qubits - 1 - gate.targ
            self._algorithm.HGate_matrixdot(
                t_index,
                matrix,
                self._vector,
                self._qubits,
                self._sync
            )
        elif gate.type() == GATE_ID["CRz"]:
            matrix = self.get_Matrix(gate)
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
        elif gate.type() == GATE_ID["Measure"]:
            index = self._qubits - 1 - gate.targ
            result = self._algorithm.MeasureGate_Measure(
                index,
                self._vector,
                self._qubits,
                self._sync
            )
            self.circuit.qubits[gate.targ].measured = result

        else:
            matrix = self.get_Matrix(gate)
            aux = cp.zeros_like(self._vector)
            self._algorithm.matrix_dot_vector(
                matrix,
                gate.controls + gate.targets,
                self._vector,
                self._qubits,
                gate.affectArgs,
                aux,
                self._sync
            )
            self.vector = aux
