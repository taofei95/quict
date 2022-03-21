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

    __PRECISION = ["single", "double"]

    def __init__(self, precision: str = "double", gpu_device_id: int = 0, sync: bool = True):
        if precision not in BasicGPUSimulator.__PRECISION:
            raise ValueError("Wrong precision. Please use one of [single, double].")

        self._precision = np.complex128 if precision == "double" else np.complex64
        self._device_id = gpu_device_id
        self._sync = sync
        self._vector = None

    def _gate_matrix_prepare(self):
        # Pretreatment gate matrixs optimizer
        self.gateM_optimizer = GateMatrixs(self._precision, self._device_id)
        self.gateM_optimizer.build(self._gates)

    @property
    def circuit(self):
        return self._circuit

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
        return self.gateM_optimizer.get_target_matrix(gate)
