"""
Synthesize a Clifford circuit unidirectionally or bidirectionally
"""

import random

import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import CompositeGate
from QuICT.core.utils.gate_type import GateType
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
    def execute(cls, gates: CompositeGate, strategy='greedy'):
        """
        Args:
            gates(Circuit/CompositeGate): the Clifford Circuit/CompositeGate to be synthesized
            strategy(str, optional): strategy of choosing qubit for each step

        Returns:
            CompositeGate: the synthesized Clifford CompositeGate
        """
        width = gates.width()
        if isinstance(gates, Circuit):
            gates = CompositeGate(gates=gates.gates)
        assert isinstance(gates, CompositeGate),\
            TypeError('Invalid input(Circuit/CompositeGate)')
        for gate in gates.gates:
            assert gate.is_clifford(), TypeError('Only Clifford gates here')
        assert strategy in ['greedy', 'random'],\
            ValueError('strategy of choosing qubit could only be "greedy" or "random"')

        def gates_next(gates: CompositeGate, disentangler: CompositeGate):
            gates_next = disentangler.inverse()
            gates_next.extend(gates)
            return gates_next

        gates_syn = CompositeGate()
        not_disentangled = list(range(width))
        if strategy == 'greedy':
            while not_disentangled:
                cnot_min = np.inf
                disentangler_min = None
                qubit_min = None
                for qubit in not_disentangled:
                    disentangler = cls.disentangle_one_qubit(gates, width, qubit)
                    if disentangler.count_2qubit_gate() < cnot_min:
                        cnot_min = disentangler.count_2qubit_gate()
                        disentangler_min = disentangler
                        qubit_min = qubit
                gates_syn.extend(disentangler_min)
                gates = gates_next(gates, disentangler_min)
                not_disentangled.remove(qubit_min)
        if strategy == 'random':
            while not_disentangled:
                qubit = random.choice(not_disentangled)
                disentangler = cls.disentangle_one_qubit(gates, width, qubit)
                gates_syn.extend(disentangler)
                gates = gates_next(gates, disentangler)
                not_disentangled.remove(qubit)

        return gates_syn

    @staticmethod
    def disentangle_one_qubit(gates: CompositeGate, width: int, target: int) -> CompositeGate:
        """
        Disentangle the target qubit from gates, i.e. for CompositeGate C, give the CompositeGate L
        such that L^-1 C acts trivially on the target qubit.

        Args:
            gates(CompositeGate): the CompositeGate to be disentangled
            width(int): the width of the operators
            target(int): the target qubit to be disentangled from gates

        Returns:
            CompositeGate: the disentangler
        """
        # Create X_j, Z_j
        x_op = [GateType.id for _ in range(width)]
        z_op = [GateType.id for _ in range(width)]
        x_op[target] = GateType.x
        z_op[target] = GateType.z
        pauli_x = PauliOperator(x_op)
        pauli_z = PauliOperator(z_op)

        # Compute C X_j C^-1 and C Z_j C^-1
        for gate in gates.inverse():
            pauli_x.conjugate_act(gate)
            pauli_z.conjugate_act(gate)

        return PauliOperator.disentangler(pauli_x, pauli_z, target)

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
