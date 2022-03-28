"""
Optimize Clifford circuits with symbolic Pauli gates
"""

from QuICT.core.gate import GateType, CLIFFORD_GATE_SET, PAULI_GATE_SET
from QuICT.qcda.optimization._optimization import Optimization


class SymbolicPeepholeOptimization(Optimization):
    """
    By decoupling CNOT gates with projectors and symbolic Pauli gates, optimization
    rules of 1-qubit gates could be used to optimize Clifford circuits.

    Reference:
        https://arxiv.org/abs/2105.02291
    """
    @classmethod
    def execute(cls):
        pass


class SymbolicPauliGate(object):
    """
    Symbolic Pauli gate gives another expression for controlled Pauli gates.

    By definition, a controlled-U gate CU means:
        if the control qubit is |0>, do nothing;
        if the control qubit is |1>, apply U to the target qubit.
    In general, CU = ∑_v |v><v| ⊗ U^v, where U^v is called a symbolic gate.
    Here we focus only on symbolic Pauli gates, symbolic phase and their
    composition, whose optimization is the core of SymbolicPeepholeOptimization.

    In this class, we use a list of GateType, each with a bool deciding if it is a symbolic gate,
    to represent the gates. Despite the gates, a symbolic phase and a phase are also recorded.
    """
    def __init__(self, gates=None, symbolic_phase=1 + 0j, phase=1 + 0j):
        """
        Construct a SymbolicPauliGate with a list of GateType and the symbolic flag

        Args:
            gates(list, optional): a list of GateType and the symbolic flag
            symbolic_phase(complex, optional): the symbolic phase of the SymbolicPauliGate, ±1 or ±i
            phase(complex, optional): the phase of the SymbolicPauliGate, ±1 or ±i
        """
        if gates is None:
            self._gates = []
        else:
            assert isinstance(gates, list), TypeError("gates must be a list.")
            for gate_type, flag in gates:
                assert gate_type == GateType.id or gate_type in CLIFFORD_GATE_SET,\
                    ValueError("gates must contain Clifford gates only.")
                if gate_type in PAULI_GATE_SET:
                    assert isinstance(flag, bool), TypeError("symbolic flag must be a bool.")
                else:
                    assert flag is False, ValueError("symbolic flag for non-Pauli gates must be False.")
            self._gates = gates
        phase_list = [1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j]
        assert symbolic_phase in phase_list, ValueError("symbolic_phase must be ±1 or ±i")
        self._symbolic_phase = symbolic_phase
        assert phase in phase_list, ValueError("phase must be ±1 or ±i")
        self._phase = phase

    @property
    def gates(self) -> list:
        return self._gates

    @gates.setter
    def gates(self, gates: list):
        assert isinstance(gates, list), TypeError("gates must be a list.")
        for gate_type, flag in gates:
            assert gate_type == GateType.id or gate_type in CLIFFORD_GATE_SET,\
                ValueError("gates must contain Clifford gates only.")
            if gate_type in PAULI_GATE_SET:
                assert isinstance(flag, bool), TypeError("symbolic flag must be a bool.")
            else:
                assert flag is False, ValueError("symbolic flag for non-Pauli gates must be False.")
        self._gates = gates

    @property
    def symbolic_phase(self) -> complex:
        return self._symbolic_phase

    @symbolic_phase.setter
    def symbolic_phase(self, symbolic_phase: complex):
        phase_list = [1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j]
        assert symbolic_phase in phase_list, ValueError("symbolic_phase must be ±1 or ±i")
        self._symbolic_phase = symbolic_phase

    @property
    def phase(self) -> complex:
        return self._phase

    @phase.setter
    def phase(self, phase: complex):
        phase_list = [1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j]
        assert phase in phase_list, ValueError("phase must be ±1 or ±i")
        self._phase = phase
