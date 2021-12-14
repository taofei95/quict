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
from QuICT.simulation.utils import option_validation, Result


class Simulator:
    """ The high-level simulation class, including CPU/GPU/Remote simulator mode.

    Args:
        device (str): The device of the simulator.
        backend (str): The backend for the simulator.
        shots (int): The running times; must be a positive integer.
        **options (dict): other optional parameters for the simulator.
    """

    __DEVICE = ["CPU", "GPU", "qiskit", "qcompute"]
    __GPU_BACKEND_MAPPING = {
        "unitary": UnitarySimulator,
        "statevector": ConstantStateVectorSimulator,
        "multiGPU": MultiStateVectorSimulator
    }
    __REMOTE_BACKEND_MAPPING = {
        "qiskit": QiskitSimulator,
        "qcompute": QuantumLeafSimulator
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

        # validated the optional parameters
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
        """ Initial CPU simulator.

        Raises:
            ValueError: Unsupportted backend
        """
        if self._backend == "statevector":
            pass
        else:
            raise ValueError(
                f"Unsupportted backend {self._backend} in cpu simulator, please using statevector."
            )

    def _load_gpu_simulator(self):
        """ Initial GPU simulator.

        Raises:
            ValueError: Unsupportted backend
        """
        if self._backend not in Simulator.__GPU_BACKEND_MAPPING.keys():
            raise ValueError(
                f"Unsupportted backend {self._backend} in gpu simulator, "
                f"please select one of [unitary, statevector, multiGPU]."
            )

        self._simulator = Simulator.__GPU_BACKEND_MAPPING[self._backend](**self._options)

    def _load_remote_simulator(self):
        """ Initial Remote simulator. """
        self._simulator = Simulator.__REMOTE_BACKEND_MAPPING[self._device](
            backend=self._backend,
            shots=self._shots,
            **self._options
        )

    def run(self, circuit: Circuit, use_previous: bool = False):
        """ start simulator with given circuit

        Args:
            circuit (Circuit): The quantum circuits.
            use_previous (bool, optional): Using the previous state vector. Defaults to False.

        Yields:
            [array]: The state vector.
        """
        result = Result(f"{self._device}-{self._backend}", self._shots, self._options)

        if self._device in Simulator.__DEVICE[2:]:
            res = self._simulator.run(circuit, use_previous)
            result.record(res["counts"])
        else:
            for _ in range(self._shots):
                _ = self._simulator.run(circuit, use_previous)

                result.record(circuit.qubits)

        return result.dumps()
