import numpy as np

from QuICT.core.circuit.circuit import Circuit
from QuICT.core.gate import BasicGate
from QuICT.core.noise import NoiseModel
from QuICT.core.operator import NoiseGate
from QuICT.core.utils import GateType, matrix_product_to_circuit
from QuICT.simulation.utils import GateSimulator
from QuICT.tools.exception.core import TypeError, ValueError
from QuICT.tools.exception.simulation import SampleBeforeRunError


class DensityMatrixSimulator:
    """ The Density Matrix Simulator

    Args:
        device (str, optional): The device type, one of [CPU, GPU]. Defaults to "CPU".
        precision (str, optional): The precision for the density matrix, one of [single, double]. Defaults to "double".
        accumulated_mode (bool): If True, calculated density matrix with Kraus Operators in NoiseGate.
            if True, p = \\sum Ki p Ki^T.conj(). Default to be False.
            [Important: set accumulated_mode for True if you need sample result, otherwise, using False for
            fast simulate time.]
    """
    def __init__(
        self,
        device: str = "CPU",
        precision: str = "double",
        accumulated_mode: bool = False
    ):
        assert device in ["CPU", "GPU"], ValueError("UnitarySimulation.device", "[CPU, GPU]", device)
        self._device = device
        assert precision in ["single", "double"], \
            ValueError("UnitarySimulation.precision", "[single, double]", precision)
        self._precision = precision
        self._dtype = np.complex128 if precision == "double" else np.complex64
        self._accumulated_mode = accumulated_mode
        self._gate_calculator = GateSimulator(self._device, self._precision)
        self._density_matrix = None

    def initial_circuit(self, circuit: Circuit, noise_model: NoiseModel):
        """ Initial the qubits, quantum gates and state vector by given quantum circuit. """
        self._origin_circuit = circuit
        self._circuit = circuit if noise_model is None else noise_model.transpile(circuit, self._accumulated_mode)
        self._noise_model = noise_model
        self._qubits = circuit.width()

        if self._precision != circuit._precision:
            circuit.set_precision(self._precision)

    def run(
        self,
        circuit: Circuit,
        density_matrix: np.ndarray = None,
        noise_model: NoiseModel = None,
        use_previous: bool = False
    ) -> np.ndarray:
        """ Simulating the given circuit through density matrix simulator.

        Args:
            circuit (Circuit): The quantum circuit.
            density_matrix (np.ndarray): The initial density matrix.
            noise_model (NoiseModel, optional): The NoiseModel contains NoiseErrors. Defaults to None.
            use_previous (bool, optional): Using the previous state vector. Defaults to False.

        Returns:
            np.ndarray: the density matrix after simulating
        """
        self.initial_circuit(circuit, noise_model)
        # Initial density matrix
        if density_matrix is not None:
            self._gate_calculator.validate_density_matrix(density_matrix)
            self._density_matrix = self._gate_calculator.normalized_matrix(density_matrix, self._qubits)
        elif (self._density_matrix is None or not use_previous):
            self._density_matrix = self._gate_calculator.get_allzero_density_matrix(self._qubits)

        self._run(self._circuit)

        # Check Readout Error in the NoiseModel
        if noise_model is not None:
            noise_model.apply_readout_error(circuit.qubits)

        return self._density_matrix

    def _run(self, noised_circuit):
        # Start simulator
        circuit = Circuit(self._qubits)
        circuit.set_precision(self._precision)
        for gate in noised_circuit.gates:
            # Store continuous BasicGates into cgate
            if isinstance(gate, BasicGate) and gate.type != GateType.measure:
                gate | circuit
                continue

            if circuit.size() > 0:
                self.apply_gates(circuit)
                circuit._gates = []

            if gate.type == GateType.measure:
                self.apply_measure(gate.targ)
            elif isinstance(gate, NoiseGate):
                self.apply_noise(gate)
            else:
                raise TypeError("DensityMatrixSimulator.run.circuit", "[BasicGate, NoiseGate]", type(gate))

        if circuit.size() > 0:
            self.apply_gates(circuit)

    def apply_gates(self, circuit: Circuit):
        """ Simulating Circuit with BasicGates

        dm = M*dm(M.conj)^T

        Args:
            cgate (CompositeGate): The CompositeGate.
        """
        cir_matrix = circuit.matrix(self._device)
        self._density_matrix = self._gate_calculator.dot(
            self._gate_calculator.dot(cir_matrix, self._density_matrix),
            cir_matrix.conj().T
        )

    def apply_noise(self, noise_gate: NoiseGate):
        """ Simulating NoiseGate.

        dm = /sum K*dm*(K.conj)^T

        Args:
            noise_gate (NoiseGate): The NoiseGate
            qubits (int): The number of qubits in the circuit.
        """
        gate_args = noise_gate.targs
        noised_matrix = self._gate_calculator.get_empty_density_matrix(self._qubits)
        for kraus_matrix in noise_gate.noise_matrix:
            umat = matrix_product_to_circuit(kraus_matrix, gate_args, self._qubits, self._device)

            noised_matrix += self._gate_calculator.dot(
                self._gate_calculator.dot(umat, self._density_matrix),
                umat.conj().T
            )

        self._density_matrix = noised_matrix.copy()

    def apply_measure(self, index: int):
        """ Simulating the MeasureGate.

        Args:
            index (int): The index of measured qubit.
        """
        _1, self._density_matrix = self._gate_calculator.apply_measure_gate_for_dm(index, self._density_matrix, self._qubits)
        self._circuit.qubits[index].measured = int(_1)

    def sample(self, shots: int) -> list:
        assert (self._density_matrix is not None), \
            SampleBeforeRunError("DensityMatrixSimulator sample without run any circuit.")
        if self._accumulated_mode or self._noise_model is None:
            original_dm = self._density_matrix.copy()

        state_list = [0] * (1 << self._qubits)
        for _ in range(shots):
            for m_id in range(self._qubits):
                self.apply_measure(m_id)

            if self._noise_model is not None:
                self._noise_model.apply_readout_error(self._circuit.qubits)

            state_list[int(self._circuit.qubits)] += 1
            if self._accumulated_mode or self._noise_model is None:
                self._density_matrix = original_dm.copy()
            else:
                self._density_matrix = self._gate_calculator.get_allzero_density_matrix(self._qubits)
                noised_circuit = self._noise_model.transpile(self._origin_circuit, self._accumulated_mode) \
                    if self._noise_model is not None else self._circuit

                self._run(noised_circuit)

        return state_list
