# !/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/12/2 2:02 下午
# @Author  : Kaiqi Li
# @File    : _gpu_simulator

import cupy as cp
import numpy as np

from QuICT.core import *
from QuICT.simulation.utils import GateMatrixs


class BasicGPUSimulator(object):
    """
    The based class for GPU simulators

    Args:
        circuit (Circuit): The quantum circuit.
        precision [np.complex64, np.complex128]: The precision for the circuit and qubits.
        gpu_device_id (int): The GPU device ID.
    """
    def __init__(self, circuit: Circuit, precision=np.complex64, gpu_device_id: int = 0):
        self._qubits = int(circuit.circuit_width())
        self._precision = precision
        self._gates = circuit.gates
        self._device_id = gpu_device_id
        self._circuit = circuit

    def _gate_matrix_prepare(self):
        # Pretreatment gate matrixs optimizer
        self.gateM_optimizer = GateMatrixs(self._precision, self._device_id)
        for gate in self._gates:
            self.gateM_optimizer.build(gate)

        self.gateM_optimizer.concentrate_gate_matrixs()

    @property
    def circuit(self):
        return self._circuit

    @circuit.setter
    def circuit(self, circuit: Circuit):
        self._circuit = circuit
        self._gates = circuit.gates
        self._qubits = int(circuit.circuit_width())

        self._gate_matrix_prepare()

    @property
    def vector(self):
        return self._vector

    @vector.setter
    def vector(self, vec):
        with cp.cuda.Device(self._device_id):
            if type(vec) is np.ndarray:
                self._vector = cp.array(vec)
            else:
                self._vector = vec

    @property
    def device(self):
        return self._device_id

    def run(self):
        pass

    def get_gate_matrix(self, gate):
        return self.gateM_optimizer.target_matrix(gate)
