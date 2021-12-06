# !/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/4/27 2:02 下午
# @Author  : Han Yu
# @File    : _simulator

import cupy as cp
import numpy as np

from QuICT.core import *
from QuICT.simulation import ConstantStateVectorSimulator, MultiStateVectorSimulator, UnitarySimulator


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
    __BACKEND = ["unitary", "statevector", "multiGPU"]

    def __init__(self, device, backend, options: dict):
        self._device = device
        self._backend = backend
        self._options = options
        self._simulator = None

        if device not in Simulator.__DEVICE:
            raise ValueError(
                f"Unsupportted device {device}, please select one of [CPU, GPU, MultiGPU]."
            )
            
        if backend not in Simulator.__BACKEND:
            raise ValueError(
                f"Unsupportted backend {backend}, please select one of [unitary, statevector, multiGPU]."
            )

        self._validate_options()

        if device == "CPU":
            self._load_cpu_simulator()
        elif device == "GPU":
            self._load_gpu_simulator()
        else:
            self._load_remote_simulator()

    def _validate_options(self):
        pass

    def _load_cpu_simulator(self):
        if self._backend == "statevector":
            pass
        else:
            raise ValueError(
                f"Unsupportted backend {self._backend} in cpu simulator, please using statevector."
            )

    def _load_gpu_simulator(self):
        if self._backend == "statevector":
            self._simulator = ConstantStateVectorSimulator(**self._options)
        elif self._backend == "multiGPU":
            self._simulator = MultiStateVectorSimulator(**self._options)
        elif self._backend == "unitary":
            self._simulator = UnitarySimulator(**self._options)
        else:
            raise ValueError(
                f"Unsupportted backend {self._backend} in gpu simulator, please select one of [unitary, statevector, multiGPU]."
            )

    def _load_remote_simulator(self):
        pass

    def run(self):
        self._simulator.run()
