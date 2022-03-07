"""
Synthesize a Clifford circuit unidirectionally or bidirectionally
"""

from QuICT.core import Circuit
from QuICT.core.gate import CompositeGate
from QuICT.qcda.synthesis._synthesis import Synthesis
from QuICT.qcda.synthesis.clifford import PauliOperator

class CliffordUnidirectionalSynthesizer(Synthesis):
    """
    Construct L_1,…,L_n such that C = L_1…L_j C_j, where C_j acts trivially on the first j qubits.
    By induction the original Clifford circuit C is synthesized.

    Reference:
        https://arxiv.org/abs/2105.02291
    """
    @classmethod
    def execute(cls, gates: CompositeGate):
        """
        Args:
            gates(Circuit/CompositeGate): the Clifford Circuit/CompositeGate to be synthesized

        Returns:
            CompositeGate: the synthesized Clifford CompositeGate
        """
        if isinstance(gates, Circuit):
            gates = CompositeGate(gates=gates.gates)
        assert isinstance(gates, CompositeGate),\
            TypeError('Invalid input(Circuit/CompositeGate)')
        for gate in gates.gates:
            assert gate.is_clifford(), TypeError('Only Clifford gates here')


class CliffordBidirectionalSynthesizer(Synthesis):
    """
    Construct L_1,…,L_n,R_1,…,R_n such that C = L_1…L_j C_j R_j…R_1,  where C_j acts trivially
    on the first j qubits.
    By induction the original Clifford circuit C is synthesized.

    Reference:
        https://arxiv.org/abs/2105.02291
    """
    @classmethod
    def execute(cls, gates: CompositeGate):
        """
        Args:
            gates(Circuit/CompositeGate): the Clifford Circuit/CompositeGate to be synthesized

        Returns:
            CompositeGate: the synthesized Clifford CompositeGate
        """
        if isinstance(gates, Circuit):
            gates = CompositeGate(gates=gates.gates)
        assert isinstance(gates, CompositeGate),\
            TypeError('Invalid input(Circuit/CompositeGate)')
        for gate in gates.gates:
            assert gate.is_clifford(), TypeError('Only Clifford gates here')
