"""
Optimize Clifford circuits with symbolic Pauli gates
"""

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

    Here we focus only on symbolic Pauli gates, symbolic phase gates and their
    composition, whose optimization is the core of SymbolicPeepholeOptimization.

    More specifically, this class contains the following parts:
    """
