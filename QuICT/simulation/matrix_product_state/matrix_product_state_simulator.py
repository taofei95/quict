import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import GateType, BasicGate, CompositeGate
from .mps_site import MPSSiteStructure


class MatrixProductStateSimulator:
    @property
    def device(self) -> str:
        return self._device

    @property
    def precision(self) -> str:
        return self._precision

    def __init__(self, device: str = "CPU", precision: str = "double"):
        """ Initial MPS Simulator

        Args:
            device (str, optional): The device type, one of [CPU, GPU]. Defaults to "CPU".
            precision (str, optional): The precision type, one of [single, double]. Defaults to "double".
        """
        self._device = device
        self._precision = precision
        self._mps = MPSSiteStructure(device, precision)

    def run(self, circuit: Circuit, quantum_state: np.ndarray = None) -> MPSSiteStructure:
        """ Start Simulator with given circuit

        Args:
            circuit (Circuit): The quantum circuits.
            quantum_state (np.ndarray, optional): The initial quantum state vector. Defaults to None.

        Returns:
            MPSSiteStructure: The Matrix Product State
        """
        qubits = circuit.width()
        self._mps.initial_mps(qubits, quantum_state)

        for gate, qindex, _ in circuit.fast_gates:
            if isinstance(gate, BasicGate):
                self._apply_basic_gate(gate, qindex)
            elif isinstance(gate, CompositeGate):
                self._apply_composite_gate(gate, qindex)
            else:
                raise ValueError("MPS Simulation only support BasicGate and CompositeGate.")

        return self._mps

    def _apply_basic_gate(self, gate: BasicGate, qindexes: list):
        if len(qindexes) == 1:
            self._mps.apply_single_gate(qindexes[0], gate.matrix)
        elif len(qindexes) == 2:
            inverse = True if gate.type in [GateType.unitary, GateType.rzx] else False
            self._mps.apply_double_gate(qindexes, gate.matrix, inverse)
        else:
            cgate = gate.build_gate()
            mapping = False if qindexes == gate.cargs + gate.targs else True
            self._apply_composite_gate(cgate, qindexes, mapping)

    def _apply_composite_gate(self, gate: CompositeGate, qindexes: list, mapping: bool = True):
        qidxes_mapping = {}
        cgate_qlist = gate.qubits
        for idx, cq in enumerate(cgate_qlist):
            qidxes_mapping[cq] = qindexes[idx]

        for cgate, cg_idx, _ in gate.fast_gates:
            real_qidx = [qidxes_mapping[idx] for idx in cg_idx] if mapping else cg_idx
            if isinstance(cgate, CompositeGate):
                self._apply_composite_gate(cgate, real_qidx)
            else:
                self._apply_basic_gate(cgate, real_qidx)
