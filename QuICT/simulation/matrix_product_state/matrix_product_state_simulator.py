import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import GateType
from .mps_site import MPSSiteStructure


class MatrixProductStateSimulator:
    # TODO: currently do not support 3-qubits gate
    def __init__(self, device: str = "CPU", precision: str = "double"):
        self._device = device
        self._precision = precision
        self._mps = None

    def run(self, circuit: Circuit, quantum_state: np.ndarray = None) -> MPSSiteStructure:
        qubits = circuit.width()
        self._mps = MPSSiteStructure(qubits, quantum_state, device=self._device, precision=self._precision)

        for gate, qindex, _ in circuit.fast_gates:
            if len(qindex) == 1:
                self._mps.apply_single_gate(qindex[0], gate.matrix)
            elif len(qindex) == 2:
                inverse = True if gate.type in [GateType.unitary, GateType.rzx] else False
                self._mps.apply_double_gate(qindex, gate.matrix, inverse)
            else:
                raise ValueError("MPS Simulation do not support 3-qubits+ quantum gates.")

        return self._mps
