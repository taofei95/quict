import numpy as np
from typing import List, Iterator
import inspect

from QuICT.core import *
from QuICT.qcda.optimization._optimization import Optimization
from QuICT.qcda.optimization.commutative_optimization import CommutativeOptimization
from QuICT.algorithm import SyntheticalUnitary
from .dag import DAG
from .phase_poly import PhasePolynomial
from .template import *
import time

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
                # print(phase_ / np.pi)
        # gates.global_phase = np.mod(gates.global_phase, 2 * np.pi)
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
        # enumerate templates and replace every occurrence
        for template in hadamard_templates:
            cnt += template.replace_all(gates) * template.weight
        return cnt

    @classmethod
    def cancel_single_qubit_gates(cls, gates: DAG):
        """
        Merge Rz gates in Clifford+Rz circuit.
        Args:
            gates(DAG): DAG of the circuit
        """
        cnt = 0
        for node in gates.topological_sort():
            # enumerate every single qubit gate
            if node.gate.qasm_name != 'rz':
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
        cls.parameterize_all(gates)

        cnt = 0
        for prev_node, succ_node, node_cnt in list(cls.enumerate_cnot_rz_circuit(gates)):
            # the boundary given by enumerate_cnot_rz_circuit is described by internal node of
            # the sub circuit, but PhasePolynomial and replace method need eternal boundary.
            # This conversion is necessary because nodes in eternal boundary may change due to previous iteration.
            for qubit_ in range(gates.width()):
                if prev_node[qubit_] is not None:
                    # TODO remove assert
                    assert succ_node[qubit_] is not None, 'internal error'
                    c_node, c_qubit = prev_node[qubit_]
                    prev_node[qubit_] = c_node.predecessors[c_qubit]
                    c_node, c_qubit = succ_node[qubit_]
                    succ_node[qubit_] = c_node.successors[c_qubit]

            # extract the sub circuit
            sub_circ = DAG.copy_sub_circuit(prev_node, succ_node)
            if DEBUG:
                print('before', 'count', node_cnt)
                sub_circ.get_circuit().draw(method='command')

            # calculate the phase poly and simplify it
            phase_poly = PhasePolynomial(sub_circ)
            circ = phase_poly.get_circuit()

            if DEBUG:
                print('after')
                circ.draw(method='command')

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

        cls.deparameterize_all(gates)
        return cnt

    @classmethod
    def float_rotations(cls, gates: DAG):
        """
        """
        print(inspect.currentframe(), 'not implemented yet')

    @classmethod
    def _execute(cls, gates, routine: List[int], verbose):
        if DEBUG:
            mat_0 = SyntheticalUnitary.run(gates)
            draw_cnt = 0
        _gates = DAG(gates)

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

                if DEBUG:
                    circ_optim = _gates.get_circuit()

                    circ_optim.draw(filename=f'{draw_cnt}.jpg')
                    draw_cnt += 1

                    mat_1 = SyntheticalUnitary.run(circ_optim)
                    if not np.allclose(mat_0, mat_1):
                        assert False
            # stop if nothing can be optimized
            if cnt == 0:
                break
            gate_cnt += cnt
        if verbose:
            print(f'{gate_cnt} / {_gates.init_size} reduced in total, cost {np.round(total_time, 3)} s')

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
