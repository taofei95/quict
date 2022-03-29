"""
Optimize Clifford circuits with template matching and symbolic peephole optimization
"""

from QuICT.core.gate import CompositeGate, GateType, H, CX
from QuICT.qcda.optimization._optimization import Optimization
from QuICT.qcda.utility import PauliOperator


class CliffordOptimization(Optimization):
    """
    Implement the Clifford circuit optimization process described in Reference, which
    consists of the following 4 steps:
    1. Partition the circuit into compute and Pauli stages
    2. Apply template matching to the compute stage
    3. Apply symbolic peephole optimization to the compute stage
    4. Optimize the 1-qubit gate count

    Reference:
        https://arxiv.org/abs/2105.02291
    """
    @classmethod
    def execute(cls, gates: CompositeGate):
        """
        Args:
            gates(Circuit/CompositeGate): the Clifford Circuit/CompositeGate to be optimized

        Returns:
            CompositeGate: the Clifford CompositeGate after optimization
        """

    @staticmethod
    def partition(gates: CompositeGate):
        """
        Partition the CompositeGate into compute and Pauli stages, where compute stage contains
        only CX, H, S, Sdg gates and Pauli stage contains only Pauli gates.

        Args:
            gates(CompositeGate): Clifford CompositeGate

        Returns:
            CompositeGate, PauliOperator: compute stage and Pauli stage
        """
        for gate in gates:
            assert gate.is_clifford(), ValueError('Only Clifford CompositeGate')
        width = gates.width()
        pauli = PauliOperator([GateType.id for _ in range(width)])

        compute = CompositeGate()
        for gate in gates:
            if gate.is_pauli():
                pauli.combine_one_gate(gate.type, gate.targ)
                continue
            if gate.type == GateType.cx or gate.type == GateType.h or gate.type == GateType.s:
                pauli.conjugate_act(gate)
                compute.append(gate.copy())
                continue
            if gate.type == GateType.sdg:
                pauli.combine_one_gate(GateType.z, gate.targ)
                pauli.conjugate_act(gate.inverse())
                compute.append(gate.inverse())
                continue

        return compute, pauli

    @staticmethod
    def symbolic_peephole_optimization(gates: CompositeGate, control_set: list):
        """
        Symbolic Pauli gate gives another expression for controlled Pauli gates.
        By definition, a controlled-U gate CU means:
            if the control qubit is |0>, do nothing;
            if the control qubit is |1>, apply U to the target qubit.
        In general, CU = ∑_v |v><v| ⊗ U^v, where U^v is called a symbolic gate.

        Here we focus only on symbolic Pauli gates and symbolic phase.
        By decoupling CNOT gates with projectors and symbolic Pauli gates, optimization
        rules of 1-qubit gates could be used to optimize Clifford circuits.

        Args:
            gates(CompositeGate): CompositeGate with CX, H, S gates only, to be optimized
            control_set(list): list of qubit, CX coupling control_set and the complement would be decoupled

        Returns:
            CompositeGate: CompositeGate after optimization
        """
        for gate in gates:
            assert gate.type in [GateType.cx, GateType.h, GateType.s],\
                ValueError('Only CX, H, S gates are allowed in this optimization')

        def HS_optimize(gates: CompositeGate):
            """
            For CompositeGate with CX, H, S only, optimize the single qubit gates trivially
            """
            gates_push = CompositeGate()
            H_stack = [[] for _ in range(gates.width())]
            S_stack = [[] for _ in range(gates.width())]
            for gate in gates:
                if gate.type == GateType.h:
                    if S_stack[gate.targ]:
                        gates_push.extend(S_stack[gate.targ])
                        S_stack[gate.targ] = []
                    H_stack[gate.targ].append(gate)
                    if len(H_stack[gate.targ]) == 2:
                        H_stack[gate.targ] = []
                if gate.type == GateType.s:
                    if H_stack[gate.targ]:
                        gates_push.extend(H_stack[gate.targ])
                        H_stack[gate.targ] = []
                    S_stack[gate.targ].append(gate)
                    if len(S_stack[gate.targ]) == 4:
                        S_stack[gate.targ] = []
                # S gates on CX.carg commutes with CX
                if gate.type == GateType.cx:
                    if H_stack[gate.carg]:
                        gates_push.extend(H_stack[gate.carg])
                        H_stack[gate.carg] = []
                    if H_stack[gate.targ]:
                        gates_push.extend(H_stack[gate.targ])
                        H_stack[gate.targ] = []
                    if S_stack[gate.targ]:
                        gates_push.extend(S_stack[gate.targ])
                        S_stack[gate.targ] = []
                    gates_push.append(gate)
            for qubit in range(gates.width()):
                gates_push.extend(H_stack[qubit])
                gates_push.extend(S_stack[qubit])
            return gates_push

        def CX_reverse(gates: CompositeGate, control_set: list):
            """
            For CX in the CompositeGate, if its target qubit is in the control_set while its control qubit
            is not, the control and target qubit will be reversed by adding H gates.
            """
            gates_reverse = CompositeGate()
            for gate in gates:
                if gate.type == GateType.cx and gate.carg not in control_set and gate.targ in control_set:
                    with gates_reverse:
                        H & gate.carg
                        H & gate.targ
                        CX & [gate.targ, gate.carg]
                        H & gate.carg
                        H & gate.targ
                else:
                    gates_reverse.append(gate)

            return gates_reverse

        gates = HS_optimize(gates)
        gates = CX_reverse(gates, control_set)
        gates = HS_optimize(gates)
        
        return gates

