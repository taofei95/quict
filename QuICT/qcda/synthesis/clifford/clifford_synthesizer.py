"""
Synthesize a Clifford circuit unidirectionally or bidirectionally
"""

import copy
import random
import multiprocessing as mp

import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import CompositeGate
from QuICT.core.utils.gate_type import GateType
from QuICT.qcda.synthesis._synthesis import Synthesis
from QuICT.qcda.utility import PauliOperator


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
    def disentangle_one_qubit(gates: CompositeGate, width: int, target: int):
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
        pauli_x = PauliOperator([GateType.id for _ in range(width)])
        pauli_z = PauliOperator([GateType.id for _ in range(width)])
        pauli_x.operator[target] = GateType.x
        pauli_z.operator[target] = GateType.z

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
    def execute(cls, gates: CompositeGate, qubit_strategy='greedy', pauli_strategy='random',
                shots=1, multiprocess=False, process=4, chunksize=64):
        """
        Args:
            gates(Circuit/CompositeGate): the Clifford Circuit/CompositeGate to be synthesized
            qubit_strategy(str, optional): strategy of choosing qubit for each step
            pauli_strategy(str, optional): strategy of choosing PauliOperator for each step
            shots(int, optional): if pauli_strategy is random, shots of random
            multiprocess(bool, optional): whether to use the multiprocessing accelaration
            process(int, optional): the number of processes in a pool
            chunksize(int, optional): iteration dealt with in a process

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
        assert qubit_strategy in ['greedy', 'random'],\
            ValueError('strategy of choosing qubit could only be "greedy" or "random"')
        assert pauli_strategy in ['brute_force', 'random'],\
            ValueError('strategy of choosing PauliOperator could only be "brute_force" or "random"')

        def insert_identity(p: PauliOperator, width: int, not_disentangled: list):
            # Operators on the disentangled qubits must be I
            for i in range(width):
                if i not in not_disentangled:
                    p.operator.insert(i, GateType.id)

        def gates_next(gates: CompositeGate, left: CompositeGate, right: CompositeGate):
            gates_next = left.inverse()
            gates_next.extend(gates)
            gates_next.extend(right)
            return gates_next

        def brute_force_iterator(gates: CompositeGate, width: int, qubit: int, not_disentangled: list):
            for p1 in PauliOperator.iterator(len(not_disentangled)):
                insert_identity(p1, width, not_disentangled)
                for p2 in PauliOperator.iterator(len(not_disentangled)):
                    insert_identity(p2, width, not_disentangled)
                    # Only anti-commutative pairs
                    if p1.commute(p2):
                        continue
                    yield gates, qubit, p1, p2

        def random_iterator(gates: CompositeGate, width: int, qubit: int, not_disentangled: list, shots: int):
            for _ in range(shots):
                p1, p2 = PauliOperator.random_anti_commutative_pair(len(not_disentangled))
                insert_identity(p1, width, not_disentangled)
                insert_identity(p2, width, not_disentangled)
                yield gates, qubit, p1, p2

        def minimum_over_pauli(gates: CompositeGate, width: int, qubit: int, not_disentangled: list):
            cnot_min = np.inf
            left_min = None
            right_min = None
            if pauli_strategy == 'brute_force':
                iterator = brute_force_iterator(gates, width, qubit, not_disentangled)
            if pauli_strategy == 'random':
                iterator = random_iterator(gates, width, qubit, not_disentangled, shots)
            if multiprocess:
                pool = mp.Pool(process)
                result = pool.starmap_async(cls.disentangle_one_qubit,
                                            iterable=iterator,
                                            chunksize=chunksize)
                for left, right in result.get():
                    if left.count_2qubit_gate() + right.count_2qubit_gate() < cnot_min:
                        cnot_min = left.count_2qubit_gate() + right.count_2qubit_gate()
                        left_min = left
                        right_min = right
                pool.close()
                pool.join()
            else:
                for gates, qubit, p1, p2 in iterator:
                    left, right = cls.disentangle_one_qubit(gates, qubit, p1, p2)
                    if left.count_2qubit_gate() + right.count_2qubit_gate() < cnot_min:
                        cnot_min = left.count_2qubit_gate() + right.count_2qubit_gate()
                        left_min = left
                        right_min = right
            return cnot_min, left_min, right_min

        gates_left = CompositeGate()
        gates_right = CompositeGate()
        not_disentangled = list(range(width))

        if qubit_strategy == 'greedy':
            while not_disentangled:
                cnot_min = np.inf
                left_min = None
                right_min = None
                qubit_min = None
                for qubit in not_disentangled:
                    cnot_cnt, left, right = minimum_over_pauli(gates, width, qubit, not_disentangled)
                    if cnot_cnt < cnot_min:
                        cnot_min = cnot_cnt
                        left_min = left
                        right_min = right
                        qubit_min = qubit
                gates_left.extend(left_min)
                gates_right.left_extend(right_min.inverse())
                gates = gates_next(gates, left_min, right_min)
                not_disentangled.remove(qubit_min)

        if qubit_strategy == 'random':
            while not_disentangled:
                qubit = random.choice(not_disentangled)
                _, left, right = minimum_over_pauli(gates, width, qubit, not_disentangled)
                gates_left.extend(left)
                gates_right.left_extend(right.inverse())
                gates = gates_next(gates, left, right)
                not_disentangled.remove(qubit)

        gates_left.extend(gates_right)
        return gates_left

    @staticmethod
    def disentangle_one_qubit(gates: CompositeGate, target: int, p1: PauliOperator, p2: PauliOperator):
        """
        Disentangle the target qubit from gates, i.e. for CompositeGate C, give the CompositeGate L, R
        such that L^-1 C R^-1 acts trivially on the target qubit.

        Args:
            gates(CompositeGate): the CompositeGate to be disentangled
            target(int): the target qubit to be disentangled from gates
            p1(PauliOperator): PauliOperator P
            p2(PauliOperator): PauliOperator P'

        Returns:
            CompositeGate, CompositeGate: the left and right disentangler
        """
        # Using the notation in the paper
        o1 = copy.deepcopy(p1)
        o2 = copy.deepcopy(p2)

        # Compute C P C^-1 and C P' C^-1
        for gate in gates.inverse():
            o1.conjugate_act(gate)
            o2.conjugate_act(gate)

        return PauliOperator.disentangler(o1, o2, target), PauliOperator.disentangler(p1, p2, target)
