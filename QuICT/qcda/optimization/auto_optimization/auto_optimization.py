import numpy as np
from typing import List
import inspect

from QuICT.core import *
from QuICT.qcda.optimization._optimization import Optimization
from QuICT.algorithm import SyntheticalUnitary
from .dag import DAG
from .phase_poly import PhasePolynomial
from .template import *


DEBUG = False


class AutoOptimization(Optimization):
    """
    Heuristic optimization of circuits in Clifford + Rz.

    [1] Nam, Yunseong, et al. "Automated optimization of large quantum
    circuits with continuous parameters." npj Quantum Information 4.1
    (2018): 1-12.
    """

    _optimize_sub_method = {
        1: "reduce_hadamard_gates",
        2: "cancel_single_qubit_gates",
        3: "cancel_two_qubit_gates",
        4: "merge_rotations",
        5: "float_rotations",
    }
    _optimize_routine = {
        'heavy': [1, 3, 2, 3, 1, 2, 5],
        'light': [1, 3, 2, 3, 1, 2, 4, 3, 2],
    }

    @classmethod
    def reduce_hadamard_gates(cls, gates: DAG):
        cnt = 0
        # enumerate templates and replace every occurrence
        for template in hadamard_templates:
            cnt += template.replace_all(gates)
        return cnt

    @classmethod
    def cancel_single_qubit_gates(cls, gates: DAG, epsilon=1e-8):
        cnt = 0
        for node in gates.topological_sort():
            # enumerate every single qubit gate
            if node.gate.qasm_name != 'rz':
                continue
            # erase the gate if degree = 0
            if abs(node.gate.parg) < epsilon:
                node.erase()
                cnt += 1
                continue

            # try cancelling while commuting the gate with templates
            # (c_node, c_qubit): the position right before template matching
            c_node, c_qubit = node, 0
            while True:
                # (n_node, n_qubit): the start of template matching
                n_node, n_qubit = c_node.successors[c_qubit]
                # stop if reaching the end of circuit
                if not n_node.gate:
                    break

                # if n_node is another rz, merge and erase the original node
                if n_node.gate.qasm_name == node.gate.qasm_name:
                    n_node.gate.pargs = n_node.gate.parg + node.gate.parg
                    node.erase()
                    cnt += 1
                    break

                # template matching
                mapping = None
                for template in single_qubit_gate_templates:
                    mapping = mapping or template.compare(c_node.successors[c_qubit])
                    if mapping:
                        # found a sub-circuit that commutes
                        # set (c_node, c_qubit) to be the last position of this sub-circuit
                        c_node, c_qubit = template.template.end_nodes[template.anchor].predecessors[0]
                        c_node = mapping[id(c_node)]
                        break
                # found no templates. commuting fails
                if not mapping:
                    break
        return cnt

    @classmethod
    def cancel_two_qubit_gates(cls, gates: DAG):
        cnt = 0
        reachable = gates.get_reachable_relation()
        for node in list(gates.topological_sort()):
            if node.flag == DAG.Node.FLAG_ERASED or node.gate.qasm_name != 'cx':
                continue

            c_ctrl_node, c_ctrl_qubit = node, 0
            c_targ_node, c_targ_qubit = node, 1
            while True:
                n_ctrl_node, n_ctrl_qubit = c_ctrl_node.successors[c_ctrl_qubit]
                n_targ_node, n_targ_qubit = c_targ_node.successors[c_targ_qubit]
                if id(n_ctrl_node) == id(n_targ_node) and n_ctrl_node.gate.qasm_name == 'cx' and \
                        n_ctrl_qubit == 0 and n_targ_qubit == 1:
                    n_ctrl_node.erase()
                    node.erase()
                    cnt += 2
                    break

                mapping = None
                for template in cnot_ctrl_template:
                    mapping = template.compare(c_ctrl_node.successors[c_ctrl_qubit])
                    if mapping and all([((id(c_targ_node), c_targ_qubit), id(o))
                                        not in reachable for o in mapping.values()]):
                        c_ctrl_node, c_ctrl_qubit = template.template.end_nodes[template.anchor].predecessors[0]
                        c_ctrl_node = mapping[id(c_ctrl_node)]
                        break
                    else:
                        mapping = None
                if mapping:
                    continue

                for template in cnot_targ_template:
                    mapping = template.compare(c_targ_node.successors[c_targ_qubit])
                    if mapping and all([((id(c_ctrl_node), c_ctrl_node), id(o))
                                        not in reachable for o in mapping.values()]):
                        c_targ_node, c_targ_qubit = template.template.end_nodes[template.anchor].predecessors[0]
                        c_targ_node = mapping[id(c_targ_node)]
                        break
                    else:
                        mapping = None
                if not mapping:
                    break
        return cnt

    @classmethod
    def enumerate_cnot_rz_circuit(cls, gates: DAG):
        gates.set_qubit_loc()
        gates.reset_flag()
        for node in gates.topological_sort():
            if node.flag == node.FLAG_VISITED or node.gate.qasm_name != 'cx':
                continue

            term_node_set = set()
            visited_cnot = {}
            anchors = {node.qubit_loc[0]: node, node.qubit_loc[1]: node}
            anchor_queue = deque(anchors.keys())
            while len(anchor_queue) > 0:
                anchor_qubit = anchor_queue.popleft()
                for neighbors in ['predecessors', 'successors']:
                    c_node = anchors[anchor_qubit]
                    while True:
                        p_node, p_qubit = getattr(c_node, neighbors)[c_node.qubit_id[anchor_qubit]]
                        if p_node.gate is None or p_node.gate.qasm_name not in ['cx', 'x', 'rz'] or \
                                p_node.flag == p_node.FLAG_VISITED:
                            term_node_set.add(id(p_node))
                            break

                        if p_node.gate.qasm_name == 'cx':
                            visited_cnot[id(p_node)] = [p_node, 0b00]
                            o_qubit = p_node.qubit_loc[p_qubit ^ 1]
                            if o_qubit not in anchors:
                                anchors[o_qubit] = p_node
                                anchor_queue.append(o_qubit)
                                visited_cnot[id(p_node)][1] |= 1 << (p_qubit ^ 1)

                        c_node = p_node

            cnot_queue = deque([node])
            # print('one visit')
            while len(cnot_queue) > 0:
                anchor_node = cnot_queue.popleft()
                for anchor_qubit in anchor_node.qubit_loc:
                    for neighbors in ['predecessors', 'successors']:
                        c_node = anchor_node
                        while True:
                            c_node.flag = c_node.FLAG_VISITED
                            p_node, p_qubit = getattr(c_node, neighbors)[c_node.qubit_id[anchor_qubit]]
                            if id(p_node) in term_node_set:
                                break
                            if p_node.gate.qasm_name == 'cx' and id(p_node) in visited_cnot:
                                visited_cnot[id(p_node)][1] |= 1 << p_node.qubit_id[anchor_qubit]
                                if visited_cnot[id(p_node)][1] == 0b11 and p_node.flag == p_node.FLAG_DEFAULT:
                                    p_node.flag = p_node.FLAG_IN_QUE
                                    cnot_queue.append(p_node)
                                break
                            if p_node.flag == p_node.FLAG_VISITED:
                                break
                            c_node = p_node

            for list_ in visited_cnot.values():
                if list_[1] != 0b11:
                    term_node_set.add(id(list_[0]))

            bound = {}
            visited_node = set()
            if DEBUG:
                print('hello')
            for neighbors in ['predecessors', 'successors']:
                bound[neighbors]: List[Tuple[DAG.Node, int]] = [None] * gates.size
                for anchor_qubit, anchor_node in anchors.items():
                    if id(anchor_node) in term_node_set:
                        continue
                    c_node = anchors[anchor_qubit]
                    while True:
                        if DEBUG:
                            if id(c_node) not in visited_node:
                                print(c_node.gate.qasm_name, c_node.gate.affectArgs, c_node.gate.pargs)
                        visited_node.add(id(c_node))
                        p_node, _ = getattr(c_node, neighbors)[c_node.qubit_id[anchor_qubit]]
                        if id(p_node) in term_node_set:
                            bound[neighbors][anchor_qubit] = (c_node, c_node.qubit_id[anchor_qubit])
                            break
                        c_node = p_node
            yield bound['predecessors'], bound['successors'], len(visited_node)

    @classmethod
    def merge_rotations(cls, gates: DAG):
        # TODO S, Sdg, T, Tdg can be included
        cnt = 0
        # gate_set = {'rz', 'cx', 'x'}
        for prev_node, succ_node, node_cnt in list(cls.enumerate_cnot_rz_circuit(gates)):
            for qubit_ in range(gates.size):
                if prev_node[qubit_] is not None:
                    # TODO remove assert
                    assert succ_node[qubit_] is not None, 'internal error'
                    c_node, c_qubit = prev_node[qubit_]
                    prev_node[qubit_] = c_node.predecessors[c_qubit]
                    c_node, c_qubit = succ_node[qubit_]
                    succ_node[qubit_] = c_node.successors[c_qubit]

            sub_circ = DAG.copy_sub_circuit(prev_node, succ_node)
            if DEBUG:
                print('before', 'count', node_cnt)
                sub_circ.get_circuit().draw(method='command')

            phase_poly = PhasePolynomial(sub_circ)
            circ = phase_poly.get_circuit()

            if DEBUG:
                print('after')
                circ.draw(method='command')

            assert circ.circuit_size() <= node_cnt, 'phase polynomial increases gate count'
            cnt += node_cnt - circ.circuit_size()
            replacement = DAG(circ)

            mapping = {}
            for qubit_ in range(gates.size):
                if not prev_node[qubit_] or not succ_node[qubit_]:
                    continue
                mapping[id(replacement.start_nodes[qubit_])] = prev_node[qubit_]
                mapping[id(replacement.end_nodes[qubit_])] = succ_node[qubit_]

            DAG.replace_circuit(mapping, replacement)
        # DONE calculate gate count delta
        return cnt

    @classmethod
    def float_rotations(cls, gates: DAG):
        print(inspect.currentframe(), 'not implemented yet')

    @classmethod
    def _execute(cls, gates, routine: List[int]):
        if DEBUG:
            mat_0 = SyntheticalUnitary.run(gates)
            draw_cnt = 0
        _gates = DAG(gates)

        draw_cnt = 0

        while True:
            cnt = 0
            for step in routine:
                cnt += getattr(cls, cls._optimize_sub_method[step])(_gates)

                # _gates.get_circuit().draw(method='command')
                # print('\noptim step', step, cnt)
                # circ_optim = _gates.get_circuit()
                # circ_optim.draw(filename=f'{draw_cnt}_{step}.jpg')
                # draw_cnt += 1

                if DEBUG:
                    circ_optim = _gates.get_circuit()

                    circ_optim.draw(filename=f'{draw_cnt}.jpg')

                    mat_1 = SyntheticalUnitary.run(circ_optim)
                    if not np.allclose(mat_0, mat_1):
                        assert False

            if cnt == 0:
                break

        return _gates.get_circuit()

    @classmethod
    def execute(cls, gates, mode='light'):
        """
        Heuristic optimization of circuits in Clifford + Rz.

        Args:
              gates(Circuit): Circuit to be optimized
              mode(str): Support 'light' and 'heavy' mode. See details in [1].
        Returns:
            Circuit: The CompositeGate after optimization

        [1] Nam, Yunseong, et al. "Automated optimization of large quantum
        circuits with continuous parameters." npj Quantum Information 4.1
        (2018): 1-12.
        """
        if mode in cls._optimize_routine:
            return cls._execute(gates, cls._optimize_routine[mode])
        else:
            raise Exception(f'unrecognized mode {mode}')
