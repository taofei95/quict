from typing import Dict, Tuple, List
from collections import deque

from QuICT.core import *
from .dag import DAG


class OptimizationTemplate:
    def __init__(self, template: DAG, replacement: DAG = None, anchor: int = 0):
        self.template = template
        self.replacement = replacement
        self.anchor = anchor

    # def replace(self, mapping: Dict[int, DAG.Node]):
        # assert self.replacement, "Template has no replacement"
        #
        # replacement = self.replacement.copy()
        # for qubit_ in range(self.template.size):
        #     # first node on qubit_ in template circuit
        #     t_node, t_qubit = self.template.start_nodes[qubit_].successors[0]
        #     # first node on qubit_ in replacement circuit
        #     r_node, r_qubit = replacement.start_nodes[qubit_].successors[0]
        #     # node previous to the node corresponding to t_node in original circuit
        #     p_node, p_qubit = mapping[id(t_node)].predecessors[t_qubit]
        #     # place r_node after p_node
        #     p_node.connect(p_qubit, r_qubit, r_node)
        #
        # for qubit_ in range(self.template.size):
        #     # last node on qubit_ in template circuit
        #     t_node, t_qubit = self.template.end_nodes[qubit_].predecessors[0]
        #     # last node on qubit_ in replacement circuit
        #     r_node, r_qubit = replacement.end_nodes[qubit_].predecessors[0]
        #     # node successive to the node corresponding to t_node in original circuit
        #     s_node, s_qubit = mapping[id(t_node)].successors[t_qubit]
        #     # place s_node after r_node
        #     r_node.connect(r_qubit, s_qubit, s_node)

    def compare(self, other: Tuple[DAG.Node, int], flag_enabled=False):
        return self.template.compare_circuit(other, self.anchor, flag_enabled)

    def replace(self, mapping: Dict[int, DAG.Node]):
        replacement = self.replacement.copy()
        new_mapping = {}
        for qubit_ in replacement.size():
            t_node, t_qubit = self.template.start_nodes[qubit_].successors[0]
            p_node, p_qubit = mapping[id(t_node)].predecessors[t_qubit]
            r_node = replacement.start_nodes[qubit_]
            new_mapping[id(r_node)] = (p_node, p_qubit)

            t_node, t_qubit = self.template.end_nodes[qubit_].predecessors[0]
            s_node, s_qubit = mapping[id(t_node)].successors[t_qubit]
            r_node = replacement.end_nodes[qubit_]
            new_mapping[id(r_node)] = (s_node, s_qubit)
        DAG.replace_circuit(new_mapping, replacement)

    def replace_all(self, dag: DAG):
        dag.reset_flag()

        matched = []
        for node in dag.topological_sort():
            for qubit_ in range(node.size):
                mapping = self.compare((node, qubit_), flag_enabled=True)
                if not mapping:
                    continue
                matched.append(mapping)

        for mapping in matched:
            self.replace(mapping)

        return len(matched)


def get_circuit_from_list(n_qubit, gate_list):
    circ = Circuit(n_qubit)
    for Gate_, qubit_ in gate_list:
        Gate_(qubit_) | circ
    return circ


def generate_hadamard_gate_templates() -> List[OptimizationTemplate]:
    tpl_list = [
        [1, [[H, 0], [S, 0], [H, 0]], [[S_dagger, 0], [H, 0], [S_dagger, 0]]],
        [1, [[H, 0], [S_dagger, 0], [H, 0]], [[S, 0], [H, 0], [S, 0]]],
        [2, [[H, 0], [H, 1], [CX, (0, 1)], [H, 0], [H, 1]], [CX, (1, 0)]],
        [2, [[H, 1], [S, 1], [CX, (0, 1)], [S_dagger, 1], [H, 1]], [[S_dagger, 1], [CX, (0, 1)], [S, 1]]],
        [2, [[H, 1], [S_dagger, 1], [CX, (0, 1)], [S, 1], [H, 1]], [[S, 1], [CX, (0, 1)], [S_dagger, 1]]],
    ]

    ret = []
    for n_qubit, tpl, rpl in tpl_list:
        tpl_circ = get_circuit_from_list(n_qubit, tpl)
        rpl_circ = get_circuit_from_list(n_qubit, rpl)
        ret.append(OptimizationTemplate(DAG(tpl_circ), DAG(rpl_circ)))
    return ret


def generate_single_qubit_gate_templates() -> List[OptimizationTemplate]:
    tpl_list = [
        [2, 1, [[H, 1], [CX, (0, 1)], [H, 1]]],
        [2, 1, [[CX, (0, 1)], [Rz(0), 1], [CX, (0, 1)]]],
        [2, 0, [[CX, (0, 1)]]]
    ]

    ret = []
    for n_qubit, anchor, tpl in tpl_list:
        tpl_circ = get_circuit_from_list(n_qubit, tpl)
        ret.append(OptimizationTemplate(DAG(tpl_circ), anchor=anchor))
    return ret


def generate_cnot_targ_templates() -> List[OptimizationTemplate]:
    tpl_list = [
        [2, 1, [[CX, (0, 1)]]],
        [2, 0, [[H, 0], [CX, (0, 1)], [H, 0]]],
    ]

    ret = []
    for n_qubit, anchor, tpl in tpl_list:
        tpl_circ = get_circuit_from_list(n_qubit, tpl)
        ret.append(OptimizationTemplate(DAG(tpl_circ), anchor=anchor))
    return ret


def generate_cnot_ctrl_templates() -> List[OptimizationTemplate]:
    tpl_list = [
        [2, 0, [[CX, (0, 1)]]],
    ]

    ret = []
    for n_qubit, anchor, tpl in tpl_list:
        tpl_circ = get_circuit_from_list(n_qubit, tpl)
        ret.append(OptimizationTemplate(DAG(tpl_circ), anchor=anchor))
    return ret


hadamard_templates = generate_hadamard_gate_templates()
single_qubit_gate_templates = generate_single_qubit_gate_templates()
cnot_targ_template = generate_cnot_targ_templates()
cnot_ctrl_template = generate_cnot_ctrl_templates()
