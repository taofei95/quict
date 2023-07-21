import numpy as np

from QuICT.core import Circuit
from .schmidt_decompostion import schmidt_decompose
from .mps_site import MPSSiteStructure


class MatrixProductStateSimulator:
    def __init__(self, device: str = "CPU", precision: str = "double"):
        self._device = device
        self._precision = precision
        self._mps = None

    def run(self, circuit: Circuit, quantum_state: np.ndarray = None) -> MPSSiteStructure:
        qubits = circuit.width()
        self._mps = MPSSiteStructure(qubits, quantum_state, device=self._device, precision=self._precision)

        for gate in circuit.flatten_gates(True):
            if gate.is_single():
                self._mps.apply_single_gate(gate.targ, gate.matrix)
            else:
                self._mps.apply_double_gate(gate.cargs + gate.targs, gate.matrix)

        return self._mps
