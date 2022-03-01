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
