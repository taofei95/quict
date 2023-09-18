
import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import Variable
from QuICT.simulation.utils import GateSimulator


class QAOASimulator:
    @property
    def vector(self):
        return self._vector

    @vector.setter
    def vector(self, vec):
        self._vector = self._gate_calculator.normalized_state_vector(vec, self._qubits)

    @property
    def device(self) -> str:
        return self._gate_calculator.device

    def __init__(self, device: str = "CPU", precision: str = "double"):
        self._device = device
        self._precision = precision
        self._gate_calculator = GateSimulator(device=self._device, precision=self._precision)
        self._base_vector = None    # The state vector before Variable Gate
        self._vector = None

    def _initial_circuit(self, circuit: Circuit):
        flatten_gates = circuit.gate_decomposition(False, False)
        for idx, gate, _, _ in enumerate(flatten_gates):
            if gate.variables > 0:
                break
        based_gates, vairable_gates = flatten_gates[:idx], flatten_gates[idx:]
        return based_gates, vairable_gates

    def run(
        self,
        circuit: Circuit,
        quantum_state: np.ndarray = None,
    ):
        # Run State Vector Simulator and generate the function relate with parameter gate
        # Store the parameter relationship and step mapping
        qubits = circuit.width()
        if quantum_state is not None:
            self._vector = self._gate_calculator.normalized_state_vector(quantum_state.copy(), self._qubits)
        else:
            self._vector = self._gate_calculator.get_allzero_state_vector(qubits)
        # Pre-compile Quantum Circuit
        based_gates, variable_gates = self._initial_circuit(circuit)
        self._vector_function = Expression(qubits)
        self._variable_mapping = [] # uid
        # Generate State Vector of Based Part
        for gate, qidxes, _ in based_gates:
            self._gate_calculator.apply_gate(
                gate, qidxes, self._vector, qubits,
                fp=False
            )
        self._base_vector = self._vector.copy()
        # Construct Variable Circuit Structure
        for gate, qidxes, _ in variable_gates:
            relation_idxes = self._gate_calculator.apply_gate(
                gate, qidxes, self._vector, qubits,
                fp=True
            )
            variables = []
            for parameter in gate.pargs:
                if isinstance(parameter, Variable):
                    variables.append(parameter)
                    if parameter.identity not in self._variable_mapping:
                        self._variable_mapping.append(parameter.identity)
            self._vector_function.add_operator(relation_idxes, variables, gate.type)
        return self.vector

    def forward(self, parameters: list):
        # Update the state vector by given parameters
        pass

    def backward(self):
        # Call backward one step of state vector，use gate's inverse
        pass
