"""
Optimize Clifford circuits with template matching and symbolic peephole optimization
"""

from QuICT.qcda.optimization._optimization import Optimization

class CliffordOptimization(Optimization):
    """
    Implement the Clifford circuit optimization process described in Reference, which
    consists of the following 4 steps:
    1. Partitioning the circuit into compute, swap and Pauli stages
    2. Applying template matching to the compute stage
    3. Applying symbolic peephole optimization to the compute stage
    4. Optimize the 1-qubit gate count

    Reference:
        https://arxiv.org/abs/2105.02291
    """
    @classmethod
    def execute(cls):
        pass
