"""
Optimize the given Circuit/CompositeGate by merging the adjacent gates with
the commutative relation between gates in consideration.
"""

import numpy as np

from QuICT.core.gate import *
from QuICT.qcda.optimization._optimization import Optimization

# Categories of combination
elimination = [
    GateType.h, GateType.x, GateType.y, GateType.z, GateType.cx,
    GateType.cy, GateType.cz, GateType.ch, GateType.ccx, GateType.swap
]
addition = [
    GateType.u1, GateType.rx, GateType.ry, GateType.rz, GateType.phase,
    GateType.crz, GateType.cu1, GateType.fsim, GateType.rxx, GateType.ryy, GateType.rzz
]
multiplication = [GateType.unitary]
other = [GateType.sx, GateType.sy, GateType.s, GateType.sdg, GateType.t, GateType.tdg]
not_calculated = [GateType.sw, GateType.u2, GateType.u3, GateType.cu3]


class Node(object):
    """
    (Temporary) implementation of Directed Acyclic Graph (DAG) used in this code

    TODO: Replace this part with a graph structure
    """
    def __init__(self, gate: BasicGate):
        """
        Args:
            gate(BasicGate): Gate represented by the node
            identity(bool): Whether the gate is identity (upon a global phase)
            predecessor(list[int]): Predecessors of the node
            reachable(bool): Whether this node needs to be compared with the new node
        """
        self.gate = gate
        self.identity = False
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
        if gate.type == GateType.x:
            return Rx(np.pi) & gate.targ, np.pi / 2
        if gate.type == GateType.sx:
            return Rx(np.pi / 2) & gate.targ, 0
        if gate.type == GateType.y:
            return Ry(np.pi) & gate.targ, np.pi / 2
        if gate.type == GateType.sy:
            return Ry(np.pi / 2) & gate.targ, 0
        if gate.type == GateType.z:
            return Rz(np.pi) & gate.targ, np.pi / 2
        if gate.type == GateType.s:
            return Rz(np.pi / 2) & gate.targ, np.pi / 4
        if gate.type == GateType.sdg:
            return Rz(-np.pi / 2) & gate.targ, -np.pi / 4
        if gate.type == GateType.t:
            return Rz(np.pi / 4) & gate.targ, np.pi / 8
        if gate.type == GateType.tdg:
            return Rz(-np.pi / 4) & gate.targ, -np.pi / 8
        return gate, 0

    @staticmethod
    def deparameterize(gate: BasicGate):
        """
        Deparameterize the parameterized gates if possible, as an inverse process of
        `parameterize` function.

        Be aware that gates like Rx(3*np.pi/2) would be transformed to X.SX (which would cause more gates).

        Args:
            gate(BasicGate): Gate to be transformed to its 'deparameterized' version

        Returns:
            Tuple[CompositeGate, float]: If deparameterization process is possible, the
                'deparameterized' version of the gate with the phase angle derived in
                the process will be returned. Otherwise, the `gate` itself with phase
                angle 0 will be returned.
        """
        gates_depara = CompositeGate()
        # Rx
        if gate.type == GateType.rx:
            # SX
            if np.isclose(np.mod(gate.parg, 2 * np.pi), np.pi / 2):
                if np.isclose(np.mod(gate.parg, 4 * np.pi), np.pi / 2):
                    gates_depara.append(SX & gate.targ)
                    return gates_depara, 0
                if np.isclose(np.mod(gate.parg, 4 * np.pi), 5 * np.pi / 2):
                    gates_depara.append(SX & gate.targ)
                    return gates_depara, np.pi
            # X
            if np.isclose(np.mod(gate.parg, 2 * np.pi), np.pi):
                if np.isclose(np.mod(gate.parg, 4 * np.pi), np.pi):
                    gates_depara.append(X & gate.targ)
                    return gates_depara, -np.pi / 2
                if np.isclose(np.mod(gate.parg, 4 * np.pi), 3 * np.pi):
                    gates_depara.append(X & gate.targ)
                    return gates_depara, np.pi / 2
            # X.SX
            if np.isclose(np.mod(gate.parg, 2 * np.pi), 3 * np.pi / 2):
                if np.isclose(np.mod(gate.parg, 4 * np.pi), 3 * np.pi / 2):
                    gates_depara.extend([X & gate.targ, SX & gate.targ])
                    return gates_depara, -np.pi / 2
                if np.isclose(np.mod(gate.parg, 4 * np.pi), 7 * np.pi / 2):
                    gates_depara.extend([X & gate.targ, SX & gate.targ])
                    return gates_depara, np.pi / 2
        # Ry
        if gate.type == GateType.ry:
            # SY
            if np.isclose(np.mod(gate.parg, 2 * np.pi), np.pi / 2):
                if np.isclose(np.mod(gate.parg, 4 * np.pi), np.pi / 2):
                    gates_depara.append(SY & gate.targ)
                    return gates_depara, 0
                if np.isclose(np.mod(gate.parg, 4 * np.pi), 5 * np.pi / 2):
                    gates_depara.append(SY & gate.targ)
                    return gates_depara, np.pi
            # Y
            if np.isclose(np.mod(gate.parg, 2 * np.pi), np.pi):
                if np.isclose(np.mod(gate.parg, 4 * np.pi), np.pi):
                    gates_depara.append(Y & gate.targ)
                    return gates_depara, -np.pi / 2
                if np.isclose(np.mod(gate.parg, 4 * np.pi), 3 * np.pi):
                    gates_depara.append(Y & gate.targ)
                    return gates_depara, np.pi / 2
            # Y.SY
            if np.isclose(np.mod(gate.parg, 2 * np.pi), 3 * np.pi / 2):
                if np.isclose(np.mod(gate.parg, 4 * np.pi), 3 * np.pi / 2):
                    gates_depara.extend([Y & gate.targ, SY & gate.targ])
                    return gates_depara, -np.pi / 2
                if np.isclose(np.mod(gate.parg, 4 * np.pi), 7 * np.pi / 2):
                    gates_depara.extend([Y & gate.targ, SY & gate.targ])
                    return gates_depara, np.pi / 2
        # Rz
        if gate.type == GateType.rz:
            # T
            if np.isclose(np.mod(gate.parg, 2 * np.pi), np.pi / 4):
                if np.isclose(np.mod(gate.parg, 4 * np.pi), np.pi / 4):
                    gates_depara.append(T & gate.targ)
                    return gates_depara, -np.pi / 8
                if np.isclose(np.mod(gate.parg, 4 * np.pi), 9 * np.pi / 4):
                    gates_depara.append(T & gate.targ)
                    return gates_depara, 7 * np.pi / 8
            # S
            if np.isclose(np.mod(gate.parg, 2 * np.pi), np.pi / 2):
                if np.isclose(np.mod(gate.parg, 4 * np.pi), np.pi / 2):
                    gates_depara.append(S & gate.targ)
                    return gates_depara, -np.pi / 4
                if np.isclose(np.mod(gate.parg, 4 * np.pi), 5 * np.pi / 2):
                    gates_depara.append(S & gate.targ)
                    return gates_depara, 3 * np.pi / 4
            # S.T
            if np.isclose(np.mod(gate.parg, 2 * np.pi), 3 * np.pi / 4):
                if np.isclose(np.mod(gate.parg, 4 * np.pi), 3 * np.pi / 4):
                    gates_depara.extend([S & gate.targ, T & gate.targ])
                    return gates_depara, -3 * np.pi / 8
                if np.isclose(np.mod(gate.parg, 4 * np.pi), 11 * np.pi / 4):
                    gates_depara.extend([S & gate.targ, T & gate.targ])
                    return gates_depara, 5 * np.pi / 8
            # Z
            if np.isclose(np.mod(gate.parg, 2 * np.pi), np.pi):
                if np.isclose(np.mod(gate.parg, 4 * np.pi), np.pi):
                    gates_depara.append(Z & gate.targ)
                    return gates_depara, -np.pi / 2
                if np.isclose(np.mod(gate.parg, 4 * np.pi), 3 * np.pi):
                    gates_depara.append(Z & gate.targ)
                    return gates_depara, np.pi / 2
            # Z.T
            if np.isclose(np.mod(gate.parg, 2 * np.pi), 5 * np.pi / 4):
                if np.isclose(np.mod(gate.parg, 4 * np.pi), 5 * np.pi / 4):
                    gates_depara.extend([Z & gate.targ, T & gate.targ])
                    return gates_depara, -5 * np.pi / 8
                if np.isclose(np.mod(gate.parg, 4 * np.pi), 13 * np.pi / 4):
                    gates_depara.extend([Z & gate.targ, T & gate.targ])
                    return gates_depara, 3 * np.pi / 8
            # S_dagger
            if np.isclose(np.mod(gate.parg, 2 * np.pi), 3 * np.pi / 2):
                if np.isclose(np.mod(gate.parg, 4 * np.pi), 3 * np.pi / 2):
                    gates_depara.append(S_dagger & gate.targ)
                    return gates_depara, 5 * np.pi / 4
                if np.isclose(np.mod(gate.parg, 4 * np.pi), 7 * np.pi / 2):
                    gates_depara.append(S_dagger & gate.targ)
                    return gates_depara, np.pi / 4
            # T_dagger
            if np.isclose(np.mod(gate.parg, 2 * np.pi), 7 * np.pi / 4):
                if np.isclose(np.mod(gate.parg, 4 * np.pi), 7 * np.pi / 4):
                    gates_depara.append(T_dagger & gate.targ)
                    return gates_depara, 9 * np.pi / 8
                if np.isclose(np.mod(gate.parg, 4 * np.pi), 15 * np.pi / 4):
                    gates_depara.append(T_dagger & gate.targ)
                    return gates_depara, np.pi / 8
        # Other
        gates_depara.append(gate)
        return gates_depara, 0

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
        assert gate_x.type == gate_y.type,\
            TypeError('Gates to be combined are not of the same type.')
        assert gate_x.cargs == gate_y.cargs and gate_x.targs == gate_y.targs,\
            ValueError('Gates to be combined are not operated on the same qubits in the same way.')

        if gate_x.type in elimination:
            # IDGates operating on all qubits are the same
            return ID.copy() & gate_x.targ

        if gate_x.type in addition:
            gate = gate_x.copy()
            for id_para in range(gate_x.params):
                if gate_x.type in [GateType.u1, GateType.cu1, GateType.fsim]:
                    gate.pargs[id_para] = np.mod(gate_x.pargs[id_para] + gate_y.pargs[id_para], 2 * np.pi)
                else:
                    gate.pargs[id_para] = np.mod(gate_x.pargs[id_para] + gate_y.pargs[id_para], 4 * np.pi)
            return gate

        if gate_x.type in multiplication:
            gate = gate_x.copy()
            gate.matrix = gate_y.matrix.dot(gate_x.matrix)
            return gate

        if gate_x.type in other or gate_x.type in not_calculated:
            raise ValueError('Gates to be combined could not be combined directly to a gate with the same type.')

        raise TypeError('Gate {} of unknown type encountered'.format(gate_x.name))

    @classmethod
    def execute(cls, gates, parameterization=True, deparameterization=False) -> CompositeGate:
        """
        Optimize the given Circuit/CompositeGate by merging the adjacent gates with
        the commutative relation between gates in consideration.

        WARNING: This method is implemented for Circuit/CompositeGate with BasicGates
        only (say, ComplexGates are not supported), other gates in the Circuit/
        CompositeGate may result in an exception or unexpected output.

        FIXME: Merging gates may cause the modification of commutative relation.
        In this version only the simplest (also the most common) case, i.e. the merged
        gate is identity, is handled. More specified analysis of the DAG is needed
        to deal with other cases, which is postponed until the graph structure is completed.

        Args:
            gates(Circuit/CompositeGate): Circuit/CompositeGate to be optimized
            parameterization(bool, optional): whether to use the parameterize() process
            deparameterization(bool, optional): whether to use the deparameterize() process

        Returns:
            CompositeGate: The CompositeGate after optimization
        """
        gates = CompositeGate(gates=gates.gates)
        nodes: list[Node] = []
        phase_angle = 0

        # Greedy optimizationn
        for gate in gates:
            gate: BasicGate
            # IDGate
            if gate.type == GateType.id:
                continue

            # PhaseGate
            if gate.type == GateType.phase:
                phase_angle += gate.parg
                continue

            # Remove such as Rot(0)
            if np.allclose(
                gate.matrix,
                gate.matrix[0, 0] * np.eye(1 << gate.controls + gate.targets)
            ):
                phase_angle += np.angle(gate.matrix[0, 0])
                continue

            # Preprocess: parameterization
            if parameterization:
                gate, phase = cls.parameterize(gate)
                phase_angle += phase
            new_node = Node(gate)

            # Main Procedure
            length = len(nodes)
            for prev in range(length):
                if nodes[prev].identity:
                    nodes[prev].reachable = False
                else:
                    nodes[prev].reachable = True

            combined = False
            for prev in range(length - 1, -1, -1):
                prev_node = nodes[prev]
                if prev_node.reachable:
                    prev_gate = prev_node.gate
                    # Combination of prev_gate and gate if same type
                    if (
                        prev_gate.type == gate.type and
                        prev_gate.cargs == gate.cargs and
                        prev_gate.targs == gate.targs and
                        not (gate.type in not_calculated)
                    ):
                        combined = True
                        nodes[prev].gate = cls.combine(prev_gate, gate)
                        mat = nodes[prev].gate.matrix
                        if (
                            nodes[prev].gate.type == GateType.id or
                            np.allclose(
                                mat,
                                mat[0, 0] * np.eye(1 << nodes[prev].gate.controls + nodes[prev].gate.targets)
                            )
                        ):
                            nodes[prev].identity = True
                        break

                    if not prev_gate.commutative(gate):
                        for node in prev_node.predecessor:
                            nodes[node].reachable = False
                        new_node.predecessor.add(prev)
                        new_node.predecessor = new_node.predecessor.union(prev_node.predecessor)

            if not combined:
                nodes.append(new_node)

        gates_opt = CompositeGate()
        for node in nodes:
            if node.identity or node.gate.type == GateType.phase:
                phase_angle += np.angle(node.gate.matrix[0, 0])
            elif deparameterization:
                gates_depara, phase = cls.deparameterize(node.gate)
                gates_opt.extend(gates_depara)
                phase_angle += phase
            else:
                gates_opt.append(node.gate)

        phase_angle = np.mod(phase_angle.real, 2 * np.pi)
        if not np.isclose(phase_angle, 0) and not np.isclose(phase_angle, 2 * np.pi):
            with gates_opt:
                Phase(phase_angle) & 0

        return gates_opt
