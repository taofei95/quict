# !/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/4/27 2:02 下午
# @Author  : Han Yu
# @File    : _simulator
import numpy as np
from typing import Union

from QuICT.core import Circuit
from QuICT.core.noise import NoiseModel
from QuICT.simulation.state_vector import CircuitSimulator
from QuICT.simulation.unitary import UnitarySimulator
from QuICT.simulation.density_matrix import DensityMatrixSimulation
from QuICT.simulation.utils import option_validation, Result


class Simulator:
    """ The high-level simulation class, including CPU/GPU/Remote simulator mode.

    Args:
        device (str): The device of the simulator. One of [CPU, GPU, qiskit, qcompute]
        backend (str): The backend for the simulator. One of [unitary, state_vector, density_matrix]
        shots (int): The running times; must be a positive integer, default to 1.
        circuit_record (bool): whether record circuit's qasm in output, default to False.
        amplitude_record (bool): whether record the amplitude of qubits, default to False.
        **options (dict): other optional parameters for the simulator.
    """

    __DEVICE = ["CPU", "GPU"]
    __BACKEND = ["unitary", "state_vector", "density_matrix"]

    def __init__(
            self,
            device: str = "CPU",
            backend: str = None,
            shots: int = 1,
            circuit_record: bool = False,
            amplitude_record: bool = False,
            **options
    ):
        assert device in Simulator.__DEVICE, "Device should be one of [CPU, GPU]."
        self._device = device
        if backend is not None:
            assert backend in Simulator.__BACKEND, "backend should be one of [unitary, state_vector, density_matrix]."
        self._backend = backend

        assert (shots >= 1)
        self._shots = shots
        self._options = self._validate_options(options=options)

        # Result's arguments
        self._circuit_record = circuit_record
        self._amplitude_record = amplitude_record

    @option_validation()
    def _validate_options(self, options, default_options=None):
        for key, value in options.items():
            default_options[key] = value

        return default_options

    def _load_simulator(self):
        """ Initial simulator. """
        if self._device == "GPU":
            from QuICT.simulation.state_vector import ConstantStateVectorSimulator

        if self._backend == "unitary":
            simulator = UnitarySimulator(device=self._device, **self._options)
        elif self._backend == "state_vector":
            simulator = ConstantStateVectorSimulator(**self._options) \
                if self._device == "GPU" else CircuitSimulator()
        elif self._backend == "density_matrix":     # TODO: DM.run has different input
            simulator = DensityMatrixSimulation(device=self._device, **self._options)
        else:
            raise ValueError(
                f"Unsupportted backend {self.backend}, please select one of {Simulator.__BACKEND}."
            )

        return simulator

    def run(
        self,
        circuit: Union[Circuit, np.ndarray],
        state_vector: np.ndarray = None,
        density_matrix: np.ndarray = None,
        noise_model: NoiseModel = None,
        use_previous: bool = False
    ):
        """ start simulator with given circuit

        Args:
            circuit (Circuit): The quantum circuits.
            noise_model (NoiseModel, optional): The NoiseModel only for density_matrix simulator. Defaults to None.
            use_previous (bool, optional): Using the previous state vector. Defaults to False.

        Yields:
            [array]: The state vector.
        """
        if not isinstance(circuit, Circuit):
            self._backend = "unitary"
        elif density_matrix is not None or noise_model is not None:
            self._backend = "density_matrix"

        if self._backend is None:
            self._backend = "state_vector"

        circuit_name = circuit.name if circuit is not None else ""
        result = Result(circuit_name, self._device, self._backend, self._shots, self._options)
        if self._circuit_record and circuit is not None:
            result.record_circuit(circuit)

        simulator = self._load_simulator()
        amplitude = simulator.run(circuit, density_matrix, noise_model, use_previous) if self._backend == "density_matrix" else \
            simulator.run(circuit, state_vector, use_previous)
        result.record_amplitude(amplitude, self._amplitude_record)

        sample_result = simulator.sample(self._shots)
        result.record_sample(sample_result)

        return result.__dict__()
