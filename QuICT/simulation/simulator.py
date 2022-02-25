# !/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/4/27 2:02 下午
# @Author  : Han Yu
# @File    : _simulator
import time

from QuICT.core import Circuit
# from QuICT.simulation.CPU_simulator import CircuitSimulator
from QuICT.simulation.gpu_simulator import (
    ConstantStateVectorSimulator,
    MultiDeviceSimulatorLauncher
)
from QuICT.simulation.unitary_simulator import UnitarySimulator
from QuICT.simulation.remote_simulator import QuantumLeafSimulator, QiskitSimulator
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
    __BACKEND = ["unitary", "statevector", "multiGPU"]
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
        self._options = self._validate_options(options=options)

        # initial simulator
        if device not in Simulator.__DEVICE:
            raise ValueError(
                f"Unsupportted device {device}, please select one of {Simulator.__DEVICE}."
            )

        self._simulator = self._load_simulator()

    @option_validation()
    def _validate_options(self, options, default_options=None):
        for key, value in options.items():
            default_options[key] = value

        return default_options

    def _load_simulator(self):
        """ Initial simulator. """
        if self._device in Simulator.__REMOTE_BACKEND_MAPPING.keys():
            return self._load_remote_simulator()

        if self._backend == "unitary":
            simulator = UnitarySimulator(device=self._device, **self._options)
        elif self._backend == "statevector":
            simulator = ConstantStateVectorSimulator(**self._options) \
                if self._device == "GPU" else None  # CircuitSimulator
        elif self._backend == "multiGPU":
            assert self._device == "GPU"
            simulator = MultiDeviceSimulatorLauncher(**self._options)
        else:
            raise ValueError(
                f"Unsupportted backend {self.backend}, please select one of {Simulator.__BACKEND}."
            )

        return simulator

    def _load_remote_simulator(self):
        """ Initial Remote simulator. """
        return Simulator.__REMOTE_BACKEND_MAPPING[self._device](
            backend=self._backend,
            shots=self._shots,
            **self._options
        )

    def run(
        self,
        circuit: Circuit,
        use_previous: bool = False,
        circuit_out: bool = False,
        statevector_out: bool = False
    ):
        """ start simulator with given circuit

        Args:
            circuit (Circuit): The quantum circuits.
            use_previous (bool, optional): Using the previous state vector. Defaults to False.

        Yields:
            [array]: The state vector.
        """
        result = Result(self._device, self._backend, self._shots, self._options)
        if circuit_out:
            result.record_circuit(circuit)

        if self._device in Simulator.__DEVICE[2:]:
            return self._simulator.run(circuit, use_previous)
        else:
            for shot in range(self._shots):
                s_time = time.time()
                state = self._simulator.run(circuit, use_previous)
                e_time = time.time()

                if statevector_out:
                    result.record_sv(state, shot)

                if self._backend != "multiGPU":
                    state = self._simulator.sample()

                result.record(state, e_time - s_time, len(circuit.qubits))

        return result.dumps()
