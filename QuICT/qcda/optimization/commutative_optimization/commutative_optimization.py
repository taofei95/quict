"""
Optimize the given Circuit/CompositeGate by merging the adjacent gates with
the commutative relation between gates in consideration.
"""

import numpy as np

from QuICT.algorithm import SyntheticalUnitary
from QuICT.core import *
from QuICT.qcda.optimization._optimization import Optimization

# Categories of combination
elimination = [GATE_ID['H'], GATE_ID['X'], GATE_ID['Y'], GATE_ID['Z'], GATE_ID['CX'],
    GATE_ID['CY'], GATE_ID['CZ'], GATE_ID['CH'], GATE_ID['CCX'], GATE_ID['Swap']]
addition = [GATE_ID['U1'], GATE_ID['Rx'], GATE_ID['Ry'], GATE_ID['Rz'], GATE_ID['Phase'],
    GATE_ID['CRz'], GATE_ID['CU1'], GATE_ID['FSim'], GATE_ID['Rxx'], GATE_ID['Ryy'], GATE_ID['Rzz']]
multiplication = [GATE_ID['Unitary']]
other = [GATE_ID['SX'], GATE_ID['SY'], GATE_ID['S'], GATE_ID['S_dagger'], GATE_ID['T'], GATE_ID['T_dagger']]
not_calculated = [GATE_ID['SW'], GATE_ID['U2'], GATE_ID['U3'], GATE_ID['CU3']]

class Node(object):
    def __init__(self, gate : BasicGate):
        self.gate = gate
        self.predecessor = set()
        self.reachable = True

class CommutativeOptimization(Optimization):
    """
    Optimize the given Circuit/CompositeGate by merging the adjacent gates with
    the commutative relation between gates in consideration.
    """
    @staticmethod
    def parameterize(gate: BasicGate):
        """
        In BasicGates, (X, SX), (Y, SY), (Z, S, Sdagger, T, Tdagger) could be
        'parameterized' to Rx, Ry, Rz respectively, which is helpful in the
        `combine` function.

        Args:
            gate(BasicGate): Gate to be transformed to its 'parameterized' version

        Returns:
            Tuple[BasicGate, float]: If the `gate` is listed above, its 'parameterized'
                version with the phase angle derived in the process will be returned.
                Otherwise, the `gate` itself with phase angle 0 will be returned.
        """
        if gate.type() == GATE_ID['X']:
            return Rx(np.pi).copy() & gate.targ, np.pi / 2
        if gate.type() == GATE_ID['SX']:
            return Rx(np.pi / 2).copy() & gate.targ, 0
        if gate.type() == GATE_ID['Y']:
            return Ry(np.pi).copy() & gate.targ, np.pi / 2
        if gate.type() == GATE_ID['SY']:
            return Ry(np.pi / 2).copy() & gate.targ, 0
        if gate.type() == GATE_ID['Z']:
            return Rz(np.pi).copy() & gate.targ, np.pi / 2
        if gate.type() == GATE_ID['S']:
            return Rz(np.pi / 2).copy() & gate.targ, np.pi / 4
        if gate.type() == GATE_ID['S_dagger']:
            return Rz(-np.pi / 2).copy() & gate.targ, -np.pi / 4
        if gate.type() == GATE_ID['T']:
            return Rz(np.pi / 4).copy() & gate.targ, np.pi / 8
        if gate.type() == GATE_ID['T_dagger']:
            return Rz(-np.pi / 4).copy() & gate.targ, -np.pi / 8
        return gate, 0

    @staticmethod
    def combine(gate_x: BasicGate, gate_y: BasicGate):
        """
        Combine `gate_x` and `gate_y` of the same type

        Generally, the combination could be divided into four categories:
        1. Elimination: the combined gate is ID
        2. Addition: the parameters of gates should be added
        3. Multiplication: the matrices of gates should be multiplied(i.e. UnitaryGate)
        4. Other: some special case(e.g. SS=Z) or not able to be calculated easily(e.g. U3Gate)
        In this method we would only deal with the first 3 cases, while the last case is partially
        handled by preprocessing the `parameterize` function.

        Args:
            gate_x(BasicGate): Gate to be combined
            gate_y(BasicGate): Gate to be combined

        Returns:
            BasicGate: The combined gate

        Raises:
            TypeError: If the input gates are not of the same type or unknown gate type encountered.
            ValueError: If the input gates are not operating on the same qubits in the same way
                or could not be combined directly to a gate with the same type.
        """
        assert gate_x.type() == gate_y.type(),\
            TypeError('Gates to be combined are not of the same type.')
        assert gate_x.cargs == gate_y.cargs and gate_x.targs == gate_y.targs,\
            ValueError('Gates to be combined are not operated on the same qubits in the same way.')
        
        if gate_x.type() in elimination:
            # IDGates operating on all qubits are the same
            return ID.copy() & gate_x.targ

        if gate_x.type() in addition:
            gate = gate_x.copy()
            for id_para in range(gate_x.params):
                gate.pargs[id_para] = gate_x.pargs[id_para] + gate_y.pargs[id_para]
            return gate

        if gate_x.type() in multiplication:
            gate = gate_x.copy()
            gate.matrix = gate_y.matrix.dot(gate_x.matrix)
            return gate

        if gate_x.type() in other or gate_x.type() in not_calculated:
            raise ValueError('Gates to be combined could not be combined directly to a gate with the same type.')
        
        raise TypeError('Gate {} of unknown type encountered'.format(gate_x.name))

    @classmethod
    def execute(cls, gates):
        """
        Optimize the given Circuit/CompositeGate by merging the adjacent gates with
        the commutative relation between gates in consideration.

        WARNING: This method is implemented for Circuit/CompositeGate with BasicGates
        only (say, ComplexGates are not supported), other gates in the Circuit/
        CompositeGate may result in an exception or unexpected output.

        Args:
            gates(Circuit/CompositeGate): Circuit/CompositeGate to be optimized

        Returns:
            CompositeGate: The CompositeGate after optimization
        """
        gates = CompositeGate(gates)
        nodes : list[Node] = []
        phase_angle = 0

        # Greedy optimizationn
        for gate in gates:
            gate : BasicGate
            # IDGate
            if gate.type() == GATE_ID['ID']:
                continue

            # PhaseGate
            if gate.type() == GATE_ID['Phase']:
                phase_angle += gate.parg
                continue

            # Preprocess: parameterization
            gate, phase = cls.parameterize(gate)
            phase_angle += phase
            new_node = Node(gate)

            # Main Procedure
            length = len(nodes)
            for prev in range(length):
                mat = nodes[prev].gate.matrix
                if nodes[prev].gate.type() == GATE_ID['ID']\
                or np.allclose(mat, mat[0, 0] * np.eye(2 ** nodes[prev].gate.targets)):
                    nodes[prev].reachable = False
                else:
                    nodes[prev].reachable = True
            
            combined = False
            for prev in range(length - 1, -1, -1):
                prev_node = nodes[prev]
                if prev_node.reachable:
                    prev_gate = prev_node.gate
                    # Combination of prev_gate and gate if same type
                    if prev_gate.type() == gate.type()\
                    and prev_gate.cargs == gate.cargs\
                    and prev_gate.targs == gate.targs\
                    and not gate.type() in not_calculated:
                        combined = True
                        nodes[prev].gate = cls.combine(prev_gate, gate)
                        break

                    if not prev_gate.commutative(gate):
                        for node in prev_node.predecessor:
                            nodes[node].reachable = False
                        new_node.predecessor.add(prev)
                        new_node.predecessor = new_node.predecessor.union(prev_node.predecessor)

            if not combined:
                nodes.append(new_node)

        gates_opt = CompositeGate(Phase(phase_angle) & 0)
        for node in nodes:
            gates_opt.append(node.gate)

        return gates_opt
