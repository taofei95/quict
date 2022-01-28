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
    MultiStateVectorSimulator
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
        self._simulator = None

        if device not in Simulator.__DEVICE:
            raise ValueError(
                f"Unsupportted device {device}, please select one of {Simulator.__DEVICE}."
            )

        # validated the optional parameters
        self._options = self._validate_options(options=options)

        if device in Simulator.__REMOTE_BACKEND_MAPPING.keys():
            self._load_remote_simulator()
        else:
            if backend not in Simulator.__BACKEND:
                raise ValueError(
                    f"Unsupportted backend {backend}, please select one of {Simulator.__BACKEND}."
                )
            else:
                self._load_simulator()

    @option_validation()
    def _validate_options(self, options, default_options=None):
        for key, value in options.items():
            default_options[key] = value

        return default_options

    def _load_simulator(self):
        """ Initial GPU simulator.

        Raises:
            ValueError: Unsupportted backend
        """
        if self._backend == "unitary":
            self._simulator = UnitarySimulator(device=self._device, **self._options)
        elif self._backend == "statevector":
            self._simulator = ConstantStateVectorSimulator(**self._options) \
                if self._device == "GPU" else None  # CircuitSimulator
        elif self._backend == "multiGPU":
            assert self._device == "GPU"
            self._simulator = MultiStateVectorSimulator(**self._options)

    def _load_remote_simulator(self):
        """ Initial Remote simulator. """
        self._simulator = Simulator.__REMOTE_BACKEND_MAPPING[self._device](
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
            res = self._simulator.run(circuit, use_previous)
            result.record(res["counts"])
        else:
            for shot in range(self._shots):
                s_time = time.time()
                state = self._simulator.run(circuit, use_previous)
                e_time = time.time()

                if statevector_out and self._backend == "statevector":
                    result.record_sv(state, shot)

                final_state = state if self._backend == "unitary" else self._simulator.sample()
                result.record(final_state, e_time - s_time, len(circuit.qubits))

        return result.dumps()
