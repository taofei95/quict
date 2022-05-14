from itertools import chain

import numpy as np
from typing import List, Iterator
import inspect

from QuICT.core import *
from QuICT.qcda.optimization._optimization import Optimization
from QuICT.qcda.optimization.commutative_optimization import CommutativeOptimization
# from QuICT.utility.decorators import deprecated
from QuICT.algorithm import SyntheticalUnitary
from .dag import DAG
from .phase_poly import PhasePolynomial
from .template import *
import time

DEBUG = False


def d_print(*args):
    if DEBUG:
        print(*args)


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
        4: "merge_rotations_upd",
        5: "float_rotations",
    }
    _optimize_routine = {
        'heavy': [1, 3, 2, 3, 1, 2, 5],
        'light': [1, 3, 2, 3, 1, 2, 4, 3, 2],
    }

    @classmethod
    def parameterize_all(cls, gates: DAG):
        for node in gates.topological_sort():
            if node.gate.qasm_name in ['s', 't', 'sdg', 'tdg', 'z']:
                gate_, phase_ = CommutativeOptimization.parameterize(node.gate)
                node.gate = gate_
                gates.global_phase += phase_

        gates.global_phase = np.mod(gates.global_phase, 2 * np.pi)
        return 0

    @classmethod
    def deparameterize_all(cls, gates: DAG):
        for node in gates.topological_sort():
            if node.gate.qasm_name == 'rz':
                compo_gate_, phase_ = CommutativeOptimization.deparameterize(node.gate)
                if len(compo_gate_.gates) > 1:
                    continue
                node.gate = compo_gate_[0]
                gates.global_phase += phase_

        gates.global_phase = np.mod(gates.global_phase, 2 * np.pi)
        return 0

    @classmethod
    def decompose_ccz_gates(cls, gates: DAG):
        pass

    @classmethod
    def reduce_hadamard_gates(cls, gates: DAG):
        """
        Reduce hadamard gates in Clifford+Rz circuits by template matching.
        Adjacent X gates will also be cancelled out.

        Args:
            gates(DAG): DAG of the circuit
        """
        cnt = 0
        cls.deparameterize_all(gates)
        # enumerate templates and replace every occurrence
        for template in hadamard_templates:
            cnt += template.replace_all(gates) * template.weight
        cls.parameterize_all(gates)
        return cnt

    @classmethod
    def cancel_single_qubit_gates(cls, gates: DAG):
        """
        Merge Rz gates in Clifford+Rz circuit.
        Args:
            gates(DAG): DAG of the circuit
        """
        cnt = 0
        for node in list(gates.topological_sort()):
            # enumerate every single qubit gate
            if node.gate.qasm_name != 'rz' or node.flag == node.FLAG_ERASED:
                continue
            # erase the gate if degree = 0
            if np.isclose(node.gate.parg, 0):
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
        """
        Merge CNOT gate in Clifford+Rz circuit.
        Args:
            gates(DAG): DAG of the circuit
        """
        cnt = 0
        reachable = gates.get_reachable_relation()
        for node in list(gates.topological_sort()):
            # enumerate every cnot gate
            if node.flag == DAG.Node.FLAG_ERASED or node.gate.qasm_name != 'cx':
                continue

            c_ctrl_node, c_ctrl_qubit = node, 0
            c_targ_node, c_targ_qubit = node, 1
            while True:
                n_ctrl_node, n_ctrl_qubit = c_ctrl_node.successors[c_ctrl_qubit]
                n_targ_node, n_targ_qubit = c_targ_node.successors[c_targ_qubit]
                # remove two adjacent cnot gates
                if id(n_ctrl_node) == id(n_targ_node) and n_ctrl_node.gate.qasm_name == 'cx' and \
                        n_ctrl_qubit == 0 and n_targ_qubit == 1:
                    n_ctrl_node.erase()
                    node.erase()
                    cnt += 2
                    break

                # try commuting node with templates anchored in control qubit
                mapping = None
                for template in cnot_ctrl_template:
                    mapping = template.compare(c_ctrl_node.successors[c_ctrl_qubit])
                    #        .___.         .___.
                    # --O----| U |--     --| U |---O--
                    #   |    |___|     =   |___|   |
                    # --X-----------     ----------X--
                    # if target node can reach any node in the template, it will block commuting:
                    #        .___.         .___.
                    # --O----|   |--     --|   |---O--
                    #   |    | U |    !=   | U |   |
                    # --X----|___|--     --|___|---X--

                    if mapping and all([((id(c_targ_node), c_targ_qubit), id(o))
                                        not in reachable for o in mapping.values()]):
                        c_ctrl_node, c_ctrl_qubit = template.template.end_nodes[template.anchor].predecessors[0]
                        c_ctrl_node = mapping[id(c_ctrl_node)]
                        break
                    else:
                        mapping = None
                if mapping:
                    continue

                # try commuting node with templates anchored in target qubit
                for template in cnot_targ_template:
                    mapping = template.compare(c_targ_node.successors[c_targ_qubit])
                    # if control node can reach any node in the template, it will block commuting.
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
    def traverse_cnot_rz_circuit(cls, anchor, flag_ofs=1):
        d_print('=== enter traverse_cnot_rz_circuit ===')

        flag_unvisited = 0
        flag_visited = flag_ofs + 1
        flag_term = flag_ofs + 2
        # flag_skipped = flag_ofs + 2

        anchors = {anchor.qubit_loc[0]: anchor, anchor.qubit_loc[1]: anchor}
        anchor_queue = deque(anchors.keys())

        term_node = {'predecessors': {}, 'successors': {}}
        d_print('first traverse')
        while len(anchor_queue) > 0:
            anchor_qubit = anchor_queue.popleft()
            d_print(f'\tanchor_q = {anchor_qubit}')
            for neighbors in ['predecessors', 'successors']:
                d_print(f'\t\tneighbors = {neighbors}')
                c_node = anchors[anchor_qubit]
                c_node.qubit_flag[c_node.qubit_id[anchor_qubit]] = flag_visited
                while True:
                    d_print(f'\t\tc_node = {c_node.gate}')
                    p_node, p_qubit = getattr(c_node, neighbors)[c_node.qubit_id[anchor_qubit]]
                    if p_node.gate is None or p_node.gate.qasm_name not in ['cx', 'x', 'rz']:
                        term_node[neighbors][anchor_qubit] = p_node
                        p_node.qubit_flag = [flag_term] * p_node.size
                        break
                    p_node.qubit_flag[p_qubit] = flag_visited

                    if p_node.gate.qasm_name == 'cx':
                        o_qubit = p_node.qubit_loc[p_qubit ^ 1]
                        if o_qubit not in anchors:
                            anchors[o_qubit] = p_node
                            anchor_queue.append(o_qubit)
                    c_node = p_node

        # left_bound = {}
        d_print('second traverse')
        for neighbors in ['predecessors', 'successors']:
            d_print(f'\tneighbors = {neighbors}')
            prune_queue = deque()
            for anchor_qubit, anchor_node in anchors.items():
                d_print(f'\t\tanchor_q = {anchor_qubit}')
                c_node = anchors[anchor_qubit]
                # c_node.qubit_flag[c_node.qubit_id[anchor_qubit]] = flag_visited

                while True:
                    d_print(f'\t\tc_node = {c_node.gate}')
                    p_node, p_qubit = getattr(c_node, neighbors)[c_node.qubit_id[anchor_qubit]]
                    if p_node.qubit_flag[p_qubit] == flag_term:
                        d_print(f'\t\t TERM: {p_node.gate}, {p_qubit}')
                        if id(p_node) != id(term_node[neighbors][anchor_qubit]):
                            prune_queue.append((p_node, p_qubit))
                        # if anchor_qubit not in left_bound and neighbors == 'predecessors':
                        #     left_bound[anchor_qubit] = (p_node, p_qubit)
                        break

                    # if p_node.qubit_flag[p_qubit] != flag_visited or p_node.gate.qasm_name != 'cx':
                    #     c_node = p_node
                    #     continue
                    if p_node.gate.qasm_name == 'cx':
                        o_qubit = p_qubit ^ 1
                        if p_node.qubit_flag[o_qubit] != flag_visited:
                            # TODO remove assert
                            # assert p_node.qubit_flag[o_qubit] != flag_term, 'internal error'
                            if o_qubit == 1:  # target out of bound, skip
                                # p_node.qubit_flag[p_qubit] = flag_skipped
                                pass
                            else:  # control out of bound, terminate
                                # print(p_node.gate, p_qubit)
                                d_print(f'\tPRUNE: {p_node.gate}, {p_qubit}')
                                p_node.qubit_flag[p_qubit] = flag_term
                                prune_queue.append((p_node, p_qubit))
                                break
                                # if neighbors == 'predecessors':
                                #     left_bound[anchor_qubit] = (p_node, p_qubit)
                    c_node = p_node

            while len(prune_queue) > 0:
                d_print('\t<< one prune step >>')
                c_node, c_qubit = prune_queue.popleft()
                while True:
                    d_print(f'\t\tc_node = {c_node.gate}')
                    p_node, p_qubit = getattr(c_node, neighbors)[c_qubit]
                    if id(p_node) == id(term_node[neighbors][p_node.qubit_loc[p_qubit]]):
                        break
                    p_node.qubit_flag[p_qubit] = flag_term
                    if p_node.gate is not None and p_node.gate.qasm_name == 'cx':
                        o_qubit = p_qubit ^ 1
                        if p_node.qubit_flag[o_qubit] == flag_visited:
                            if o_qubit == 0:
                                pass
                                # p_node.qubit_flag[o_qubit] = flag_skipped
                            else:
                                p_node.qubit_flag[o_qubit] = flag_term
                                prune_queue.append((p_node, o_qubit))
                    c_node, c_qubit = p_node, p_qubit

                    # else:  # prune unreachable node
                    #     p_node.qubit_flag[p_qubit] = p_node.FLAG_DEFAULT
                    #     if p_node.gate is not None and p_node.gate.qasm_name == 'cx':
                    #         if p_qubit == 0:
                    #             p_node.qubit_flag[p_qubit ^ 1] = flag_ter_
                    #         else:
                    #             p_node.qubit_flag[p_qubit ^ 1] = p_node.FLAG_DEFAULT

        left_bound = {}
        anchors = {anchor.qubit_loc[0]: anchor, anchor.qubit_loc[1]: anchor}
        anchor_queue = deque(anchors.keys())
        d_print('third traverse')
        while len(anchor_queue) > 0:
            anchor_qubit = anchor_queue.popleft()
            d_print(f'\tanchor_q = {anchor_qubit}')
            for neighbors in ['predecessors', 'successors']:
                d_print(f'\t\tneighbors = {neighbors}')
                c_node = anchors[anchor_qubit]
                while True:
                    d_print(f'\t\tc_node = {c_node.gate}')
                    p_node, p_qubit = getattr(c_node, neighbors)[c_node.qubit_id[anchor_qubit]]
                    if p_node.qubit_flag[p_qubit] == flag_term:
                        if neighbors == 'predecessors':
                            left_bound[anchor_qubit] = (p_node, p_qubit)
                        break
                    if p_node.gate.qasm_name == 'cx' and all([f == flag_visited for f in p_node.qubit_flag]):
                        o_qubit = p_node.qubit_loc[p_qubit ^ 1]
                        if o_qubit not in anchors:
                            anchors[o_qubit] = p_node
                            anchor_queue.append(o_qubit)
                    c_node = p_node

        edge_count = {}
        queue = deque()
        for node_, qubit_ in left_bound.values():
            s_node, s_qubit = node_.successors[qubit_]
            if s_node.qubit_flag[s_qubit] != flag_term:
                if id(s_node) not in edge_count:
                    edge_count[id(s_node)] = sum([f == flag_visited for f in s_node.qubit_flag])
                    # o_qubit = s_qubit ^ 1
                    # if s_node.gate.qasm_name == 'cx' and s_node.qubit_flag[o_qubit] != flag_vis_:
                    #     edge_count[id(s_node)] -= 1
                edge_count[id(s_node)] -= 1
                if edge_count[id(s_node)] == 0:
                    # d_print('---', node_.gate, '->', s_node.gate)
                    queue.append(s_node)

        d_print('get sub circuit')
        node_list = []
        while len(queue) > 0:
            cur = queue.popleft()
            if cur.gate is not None and all([f == flag_visited for f in cur.qubit_flag]):
                cur.flag = cur.FLAG_VISITED
                node_list.append(cur)
                d_print(f'\tcur = {cur.gate}')
            for c_qubit in range(cur.size):
                if cur.qubit_flag[c_qubit] != flag_visited:
                    continue
                s_node, s_qubit = cur.successors[c_qubit]
                if s_node is None or s_node.qubit_flag[s_qubit] == flag_term:
                    continue
                if id(s_node) not in edge_count:
                    edge_count[id(s_node)] = sum([f == flag_visited for f in s_node.qubit_flag])
                    # edge_count[id(s_node)] = s_node.size
                    # o_qubit = s_qubit ^ 1
                    # if s_node.gate.qasm_name == 'cx' and s_node.qubit_flag[o_qubit] != flag_vis_:
                    #     edge_count[id(s_node)] -= 1
                edge_count[id(s_node)] -= 1
                if edge_count[id(s_node)] == 0:
                    # d_print('---', cur.gate, '->', s_node.gate)
                    queue.append(s_node)
        return node_list

    @classmethod
    def parse_cnot_rz_circuit(cls, node_list):
        phases = {}
        first_rz = {}
        cur_phases = {}

        for node_ in node_list:
            # if node_.flag != node_.FLAG_IN_QUE:
            #     continue
            # node_.flag = node_.FLAG_VISITED

            gate_ = node_.gate
            for qubit_ in chain(gate_.cargs, gate_.targs):
                if qubit_ not in cur_phases:
                    cur_phases[qubit_] = 1 << (qubit_ + 1)

            if gate_.qasm_name == 'cx':
                cur_phases[gate_.targ] = cur_phases[gate_.targ] ^ cur_phases[gate_.carg]
            elif gate_.qasm_name == 'x':
                cur_phases[gate_.targ] = cur_phases[gate_.targ] ^ 1
            elif gate_.qasm_name == 'rz':
                sign = -1 if cur_phases[gate_.targ] & 1 else 1
                mono = (cur_phases[gate_.targ] >> 1)
                phases[mono] = sign * gate_.parg + (phases[mono] if mono in phases else 0)
                if mono not in first_rz:
                    first_rz[mono] = (node_, sign)
                else:
                    node_.flag = node_.FLAG_TO_ERASE

        for phase_, pack_ in first_rz.items():
            node_, sign = pack_
            if np.isclose(phases[phase_], 0):
                node_.erase()
            else:
                node_.gate.pargs = sign * phases[phase_]

        cnt = 0
        for node_ in node_list:
            if node_.gate.qasm_name != 'rz' or node_.flag == node_.FLAG_ERASED:
                continue
            if node_.flag == node_.FLAG_TO_ERASE:
                cnt += 1
                node_.erase()
        return cnt

    @classmethod
    def merge_rotations_upd(cls, gates: DAG):
        if DEBUG:
            gates.get_circuit().draw(filename='before_merge.jpg')
        gates.reset_flag()
        gates.set_qubit_loc()
        cnt = 0
        for idx, anchor_ in enumerate(list(gates.topological_sort())):
            if anchor_.gate.qasm_name != 'cx' or anchor_.flag == anchor_.FLAG_VISITED:
                continue

            # mat_1 = SyntheticalUnitary.run(gates.get_circuit())
            node_list = cls.traverse_cnot_rz_circuit(anchor_, idx * 2)
            cnt += cls.parse_cnot_rz_circuit(node_list)

            # mat_2 = SyntheticalUnitary.run(gates.get_circuit())
            # assert np.allclose(mat_1, mat_2), 'mat_1 != mat_2'

        if DEBUG:
            gates.get_circuit().draw(filename='after_merge.jpg')
        return cnt

    @classmethod
    def enumerate_cnot_rz_circuit(cls, gates: DAG):
        """
        Iterate over CNOT+Rz sub circuit in Clifford+Rz circuit.
        Isolated Rz gates will be ignored. Each sub circuit is
        described by a tuple (left boundary, right boundary, node count). The boundary is described
        by a list whose i-th element is the tuple (node, wire) on the i-th qubit.

        Args:
            gates(DAG): DAG of the circuit

        Returns:
            Iterator[Tuple[List[DAG.Node, int]], Tuple[List[DAG.Node, int]], int]: CNOT+Rz Sub circuit
        """

        # TODO update qubit location online
        # reset qubit location of nodes in case it is corrupted sub circuit replacement
        gates.set_qubit_loc()
        # reset node flags to FLAG_DEFAULT
        gates.reset_flag()
        for node in gates.topological_sort():
            # enumerate over every unvisited CNOT gate
            if node.flag == node.FLAG_VISITED or node.gate.qasm_name != 'cx':
                continue

            # set of terminate node's id
            term_node_set = set()
            # encountered CNOT gates during this iteration
            # id -> Tuple[node, bitmask], bitmask(i) represents if node's i-th qubit is reachable
            visited_cnot = {}

            # STEP 1: for each anchored qubit, traverse to the left and right boundary
            anchors = {node.qubit_loc[0]: node, node.qubit_loc[1]: node}
            anchor_queue = deque(anchors.keys())
            while len(anchor_queue) > 0:
                anchor_qubit = anchor_queue.popleft()
                for neighbors in ['predecessors', 'successors']:
                    c_node = anchors[anchor_qubit]
                    while True:
                        p_node, p_qubit = getattr(c_node, neighbors)[c_node.qubit_id[anchor_qubit]]

                        # find a non CNOT+Rz gate. Set it a terminate node
                        if p_node.gate is None or p_node.gate.qasm_name not in ['cx', 'x', 'rz'] or \
                                p_node.flag == p_node.FLAG_VISITED:
                            term_node_set.add(id(p_node))
                            break

                        if p_node.gate.qasm_name == 'cx':
                            # encounter a new CNOT gate
                            visited_cnot[id(p_node)] = [p_node, 0b00]
                            # if we encounter a CNOT acting on a new qubit, set it as an anchor node
                            o_qubit = p_node.qubit_loc[p_qubit ^ 1]
                            if o_qubit not in anchors:
                                anchors[o_qubit] = p_node
                                anchor_queue.append(o_qubit)
                                # anchor qubit is regarded as initially reachable
                                visited_cnot[id(p_node)][1] |= 1 << (p_qubit ^ 1)

                        c_node = p_node

            # STEP 2: topologically visit the CNOT node in visited_cnot. Calculate reachable bitmasks.
            cnot_queue = deque([node])
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

            # If any of the two qubits of one CNOT is not reachable, it cannot be included
            # in the graph. Set it as terminate node.
            for list_ in visited_cnot.values():
                if list_[1] != 0b11:
                    term_node_set.add(id(list_[0]))

            # STEP 3: calculate boundary by traversing each qubit until we encounter a terminate node
            bound = {}
            visited_node = set()
            for neighbors in ['predecessors', 'successors']:
                bound[neighbors]: List[Tuple[DAG.Node, int]] = [None] * gates.width()
                for anchor_qubit, anchor_node in anchors.items():
                    if id(anchor_node) in term_node_set:
                        continue
                    c_node = anchors[anchor_qubit]
                    while True:
                        visited_node.add(id(c_node))
                        p_node, _ = getattr(c_node, neighbors)[c_node.qubit_id[anchor_qubit]]
                        if id(p_node) in term_node_set:
                            bound[neighbors][anchor_qubit] = (c_node, c_node.qubit_id[anchor_qubit])
                            break
                        c_node = p_node
            yield bound['predecessors'], bound['successors'], len(visited_node)

    @classmethod
    def merge_rotations(cls, gates: DAG):
        cnt = 0
        for prev_node, succ_node, node_cnt in list(cls.enumerate_cnot_rz_circuit(gates)):
            # the boundary given by enumerate_cnot_rz_circuit is described by internal node of
            # the sub circuit, but PhasePolynomial and replace method need eternal boundary.
            # This conversion is necessary because nodes in eternal boundary may change due to previous iteration.
            for qubit_ in range(gates.width()):
                if prev_node[qubit_] is not None:
                    c_node, c_qubit = prev_node[qubit_]
                    prev_node[qubit_] = c_node.predecessors[c_qubit]
                    c_node, c_qubit = succ_node[qubit_]
                    succ_node[qubit_] = c_node.successors[c_qubit]

            # extract the sub circuit
            sub_circ = DAG.copy_sub_circuit(prev_node, succ_node)

            # calculate the phase poly and simplify it
            phase_poly = PhasePolynomial(sub_circ)
            circ = phase_poly.get_circuit()

            assert circ.size() <= node_cnt, 'phase polynomial increases gate count'
            cnt += node_cnt - circ.size()
            replacement = DAG(circ)

            # calculate node mapping: replacement -> circuit
            mapping = {}
            for qubit_ in range(gates.width()):
                if not prev_node[qubit_] or not succ_node[qubit_]:
                    continue
                mapping[id(replacement.start_nodes[qubit_])] = prev_node[qubit_]
                mapping[id(replacement.end_nodes[qubit_])] = succ_node[qubit_]

            DAG.replace_circuit(mapping, replacement)
            sub_circ.destroy()

        return cnt

    @classmethod
    def float_rotations(cls, gates: DAG):
        """
        """
        print(inspect.currentframe(), 'not implemented yet')

    @classmethod
    def _execute(cls, gates, routine: List[int], verbose):
        _gates = DAG(gates)
        # _gates.get_circuit().draw(filename='decompose.jpg')
        cls.parameterize_all(_gates)

        gate_cnt = 0
        round_cnt = 0
        total_time = 0
        while True:
            round_cnt += 1
            if verbose:
                print(f'ROUND #{round_cnt}:')

            cnt = 0
            # apply each method in routine
            for step in routine:

                start_time = time.time()
                cur_cnt = getattr(cls, cls._optimize_sub_method[step])(_gates)
                end_time = time.time()

                if verbose:
                    print(f'\t{cls._optimize_sub_method[step]}: {cur_cnt} '
                          f'gates reduced, cost {np.round(end_time - start_time, 3)} s')

                cnt += cur_cnt
                total_time += end_time - start_time

                # if step == 4:
                #     return _gates.get_circuit()

            # stop if nothing can be optimized
            if cnt == 0:
                break
            gate_cnt += cnt
        if verbose:
            print(f'{gate_cnt} / {_gates.init_size} reduced in total, '
                  f'remain {_gates.init_size - gate_cnt} gates, cost {np.round(total_time, 3)} s')

        cls.deparameterize_all(_gates)
        return _gates.get_circuit()

    @classmethod
    def execute(cls, gates, mode='light', verbose=False):
        """
        Heuristic optimization of circuits in Clifford + Rz.

        Args:
              gates(Circuit): Circuit to be optimized
              mode(str): Support 'light' and 'heavy' mode. See details in [1]
              verbose(bool): whether output details of each step
        Returns:
            Circuit: The CompositeGate after optimization

        [1] Nam, Yunseong, et al. "Automated optimization of large quantum
        circuits with continuous parameters." npj Quantum Information 4.1
        (2018): 1-12.
        """
        if mode in cls._optimize_routine:
            return cls._execute(gates, cls._optimize_routine[mode], verbose)
        else:
            raise Exception(f'unrecognized mode {mode}')
