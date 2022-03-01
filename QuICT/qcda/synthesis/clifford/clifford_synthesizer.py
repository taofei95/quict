"""
Synthesize a Clifford circuit unidirectionally or bidirectionally
"""

from QuICT.qcda.synthesis._synthesis import Synthesis

class CliffordUnidirectionalSynthesizer(Synthesis):
    """
    Construct L_1,…,L_n such that C = L_1…L_j C_j, where C_j acts trivially on the first j qubits.
    By induction the original Clifford circuit C is synthesized.

    Reference:
        https://arxiv.org/abs/2105.02291
    """
    @classmethod
    def execute(cls):
        pass


class CliffordBidirectionalSynthesizer(Synthesis):
    """
    Construct L_1,…,L_n,R_1,…,R_n such that C = L_1…L_j C_j R_j…R_1,  where C_j acts trivially
    on the first j qubits.
    By induction the original Clifford circuit C is synthesized.

    Reference:
        https://arxiv.org/abs/2105.02291
    """
    @classmethod
    def execute(cls):
        pass
