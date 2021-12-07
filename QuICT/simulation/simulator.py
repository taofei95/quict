# !/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/4/27 2:02 下午
# @Author  : Han Yu
# @File    : _simulator

import cupy as cp
import numpy as np

from QuICT.core import *
from QuICT.simulation import ConstantStateVectorSimulator, MultiStateVectorSimulator, UnitarySimulator
from QuICT.simulation.remote_simulator.RemoteAgent import QuantumLeafAgent, QiskitAgent


"""
device: [CPU, GPU, Multi-GPU, Remote]
backend:   [CPU - [statevector]
            GPU - [statevector, unitary]
            Multi-GPU - [statevector]
            Remote - [...]]
parameters: parameters check; using decorator?

Raises:
    ValueError: [description]
"""


class Simulator:
    __DEVICE = ["CPU", "GPU", "Remote"]
    __GPU_BACKEND_MAPPING = {
        "unitary": UnitarySimulator,
        "statevector": ConstantStateVectorSimulator,
        "multiGPU": MultiStateVectorSimulator
    }

    def __init__(
        self,
        device: str,
        backend: str,
        shots: int,
        **options
    ):
        self._device = device
        self._backend = backend
        self._shots = shots
        self._simulator = None

        if device not in Simulator.__DEVICE:
            raise ValueError(
                f"Unsupportted device {device}, please select one of [CPU, GPU, MultiGPU]."
            )

        self._options = self._validate_options(options)

        if device == "CPU":
            self._load_cpu_simulator()
        elif device == "GPU":
            self._load_gpu_simulator()
        else:
            self._load_remote_simulator()

    def _validate_options(self, options, default_options=None):
        pass

    def _load_cpu_simulator(self):
        if self._backend == "statevector":
            pass
        else:
            raise ValueError(
                f"Unsupportted backend {self._backend} in cpu simulator, please using statevector."
            )

    def _load_gpu_simulator(self):
        if self._backend not in Simulator.__GPU_BACKEND_MAPPING.keys():
            raise ValueError(
                f"Unsupportted backend {self._backend} in gpu simulator, "
                f"please select one of [unitary, statevector, multiGPU]."
            )

        self._simulator = Simulator.__GPU_BACKEND_MAPPING[self._backend](
            circuit=self._circuit,
            **self._options
        )

    def _load_remote_simulator(self):
        if self._backend in QuantumLeafAgent.__backends:
            self._simulator = QuantumLeafAgent(**self._options)
        elif self._backend in QiskitAgent.__backends:
            self._simulator = QiskitAgent(**self._options)
        else:
            raise ValueError(
                f"Unsupportted backend {self._backend} in remote simulator, please using one of "
                f"{QuantumLeafAgent.__backends} {QiskitAgent.__backends}"
            )

    def run(self, circuit: Circuit, use_previous: bool = False):
        self._simulator.run(circuit, use_previous)
