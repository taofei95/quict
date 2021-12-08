# !/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/4/27 2:02 下午
# @Author  : Han Yu
# @File    : _simulator

from QuICT.core import *
from QuICT.simulation import (
    ConstantStateVectorSimulator,
    MultiStateVectorSimulator,
    UnitarySimulator,
    QuantumLeafSimulator,
    QiskitSimulator
)
from QuICT.simulation.utils import option_validation


class Simulator:
    __DEVICE = ["CPU", "GPU", "Qiskit", "QCompute"]
    __GPU_BACKEND_MAPPING = {
        "unitary": UnitarySimulator,
        "statevector": ConstantStateVectorSimulator,
        "multiGPU": MultiStateVectorSimulator
    }

    def __init__(
        self,
        device: str,
        backend: str,
        shots: int = 1,
        **options
    ):
        assert (shots >= 1)
        self._device = device
        self._backend = backend
        self._shots = shots
        self._simulator = None

        if device not in Simulator.__DEVICE:
            raise ValueError(
                f"Unsupportted device {device}, please select one of {Simulator.__DEVICE}."
            )

        self._options = self._validate_options(options=options)

        if device == "CPU":
            self._load_cpu_simulator()
        elif device == "GPU":
            self._load_gpu_simulator()
        else:
            self._load_remote_simulator()

    @option_validation()
    def _validate_options(self, options, default_options=None):
        for key, value in options.items():
            default_options[key] = value

        return default_options

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

        self._simulator = Simulator.__GPU_BACKEND_MAPPING[self._backend](**self._options)

    def _load_remote_simulator(self):
        if self._device == "QCompute":
            self._simulator = QuantumLeafSimulator(
                backend=self._backend,
                shots=self._shots,
                **self._options
            )
        elif self._device == "Qiskit":
            self._simulator = QiskitSimulator(
                backend=self._backend,
                shots=self._shots,
                **self._options
            )
        else:
            raise ValueError(
                f"Unsupportted backend {self._backend} in remote simulator, please using one of "
                f"{Simulator.__DEVICE[2:]}."
            )

    def run(self, circuit: Circuit, use_previous: bool = False):
        if self._device in Simulator.__DEVICE[2:]:
            self._shots = 1

        for _ in range(self._shots):
            result = self._simulator.run(circuit, use_previous)

            yield result
