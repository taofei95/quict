# !/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/4/27 2:02 下午
# @Author  : Han Yu
# @File    : _simulator
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
        backend (str): The backend for the simulator. One of [unitary, statevector, density_matrix]
        shots (int): The running times; must be a positive integer, default to 1.
        circuit_record (bool): whether record circuit's qasm in output, default to False.
        amplitude_record (bool): whether record the amplitude of qubits, default to False.
        **options (dict): other optional parameters for the simulator.
    """

    __DEVICE = ["CPU", "GPU"]
    __REMOTE_DEVICE = ["qiskit", "qcompute"]
    __BACKEND = ["unitary", "statevector", "density_matrix"]

    def __init__(
            self,
            device: str,
            backend: str,
            shots: int = 1,
            circuit_record: bool = False,
            amplitude_record: bool = False,
            **options
    ):
        assert (shots >= 1)
        self._device = device
        self._backend = backend
        self._shots = shots
        self._circuit_record = circuit_record
        self._amplitude_record = amplitude_record
        self._options = self._validate_options(options=options)

        # initial simulator
        if self._device in Simulator.__DEVICE:
            self._simulator = self._load_simulator()
        elif self._device in Simulator.__REMOTE_DEVICE:
            self._simulator = self._load_remote_simulator()
        else:
            raise ValueError(
                f"Unsupportted device {device}, please select one of {Simulator.__DEVICE + Simulator.__REMOTE_DEVICE}."
            )

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
        elif self._backend == "statevector":
            simulator = ConstantStateVectorSimulator(**self._options) \
                if self._device == "GPU" else CircuitSimulator()
        elif self._backend == "density_matrix":     # TODO: DM.run has different input
            simulator = DensityMatrixSimulation(device=self._device, **self._options)
        else:
            raise ValueError(
                f"Unsupportted backend {self.backend}, please select one of {Simulator.__BACKEND}."
            )

        return simulator

    def _load_remote_simulator(self):
        """ Initial Remote simulator. """
        from QuICT.simulation.remote import QuantumLeafSimulator, QiskitSimulator

        if self._device == "qiskit":
            simulator = QiskitSimulator(backend=self._backend, shots=self._shots, **self._options)
        elif self._device == "qcompute":
            simulator = QuantumLeafSimulator(backend=self._backend, shots=self._shots, **self._options)
        else:
            raise ValueError(
                f"Unsupportted remote device {self._device}, please select one of [qiskit, qcompute]."
            )

        return simulator

    def run(
        self,
        circuit: Circuit,
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
        result = Result(circuit.name, self._device, self._backend, self._shots, self._options)
        if self._circuit_record:
            result.record_circuit(circuit)

        if self._device in Simulator.__REMOTE_DEVICE:
            return self._simulator.run(circuit, use_previous)

        amplitude = self._simulator.run(circuit, noise_model, use_previous) if self._backend == "density_matrix" else \
            self._simulator.run(circuit, use_previous)
        if self._amplitude_record:
            result.record_amplitude(amplitude)

        sample_result = self._simulator.sample(self._shots)
        result.record_sample(sample_result)

        return result
