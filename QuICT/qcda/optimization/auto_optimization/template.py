from typing import Dict, Tuple, List
from collections import deque

from QuICT.core import *
from QuICT.core.gate import *
from .dag import DAG


class OptimizationTemplate:
    def __init__(self, template: DAG, replacement: DAG = None,
                 anchor: int = 0, weight: int = 1, phase: float = 0):
        self.template = template
        self.replacement = replacement
        self.anchor = anchor
        self.weight = weight
        self.phase = phase

    def compare(self, other: Tuple[DAG.Node, int], flag_enabled=False):
        return self.template.compare_circuit(other, self.anchor, flag_enabled)

    def replace(self, mapping: Dict[int, DAG.Node]):
        replacement = self.replacement.copy()
        new_mapping = {}
        for qubit_ in range(replacement.width()):
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
            mapping = self.compare((node, -1), flag_enabled=True)
            if not mapping:
                continue
            matched.append(mapping)

        for mapping in matched:
            self.replace(mapping)

        dag.global_phase = np.mod(dag.global_phase + len(matched) * self.phase, 2 * np.pi)
        return len(matched)


def get_circuit_from_list(n_qubit, gate_list):
    circ = Circuit(n_qubit)
    for Gate_, qubit_ in gate_list:
        Gate_ | circ(qubit_)
    return circ


def generate_hadamard_gate_templates() -> List[OptimizationTemplate]:
    tpl_list = [
        [1, 2, 0, [[H, 0], [H, 0]], []],
        [1, 1, +np.pi/4, [[H, 0], [S, 0], [H, 0]], [[S_dagger, 0], [H, 0], [S_dagger, 0]]],
        [1, 1, -np.pi/4, [[H, 0], [S_dagger, 0], [H, 0]], [[S, 0], [H, 0], [S, 0]]],
        [2, 4, 0, [[H, 0], [H, 1], [CX, [0, 1]], [H, 0], [H, 1]], [[CX, [1, 0]]]],
        [2, 2, 0, [[H, 1], [S, 1], [CX, [0, 1]], [S_dagger, 1], [H, 1]], [[S_dagger, 1], [CX, [0, 1]], [S, 1]]],
        [2, 2, 0, [[H, 1], [S_dagger, 1], [CX, [0, 1]], [S, 1], [H, 1]], [[S, 1], [CX, [0, 1]], [S_dagger, 1]]],
        [1, 2, 0, [[X, 0], [X, 0]], []]
    ]

    ret = []
    for n_qubit, weight, phase, tpl, rpl in tpl_list:
        tpl_circ = get_circuit_from_list(n_qubit, tpl)
        rpl_circ = get_circuit_from_list(n_qubit, rpl)
        ret.append(OptimizationTemplate(DAG(tpl_circ), DAG(rpl_circ), weight=weight, phase=phase))
    return ret


def generate_single_qubit_gate_templates() -> List[OptimizationTemplate]:
    tpl_list = [
        [2, 1, [[H, 1], [CX, [0, 1]], [H, 1]]],
        [2, 1, [[CX, [0, 1]], [Rz(0), 1], [CX, [0, 1]]]],
        [2, 0, [[CX, [0, 1]]]],
        [3, 0, [[CX, [1, 0]], [CX, [0, 2]], [CX, [1, 0]]]],
        [1, 0, [[H, 0], [X, 0], [H, 0]]]
    ]

    ret = []
    for n_qubit, anchor, tpl in tpl_list:
        tpl_circ = get_circuit_from_list(n_qubit, tpl)
        ret.append(OptimizationTemplate(DAG(tpl_circ), anchor=anchor))
    return ret


def generate_cnot_targ_templates() -> List[OptimizationTemplate]:
    tpl_list = [
        [2, 1, [[CX, [0, 1]]]],
        [2, 0, [[H, 0], [CX, [0, 1]], [H, 0]]],
        [1, 0, [[X, 0]]]
    ]

    ret = []
    for n_qubit, anchor, tpl in tpl_list:
        tpl_circ = get_circuit_from_list(n_qubit, tpl)
        ret.append(OptimizationTemplate(DAG(tpl_circ), anchor=anchor))
    return ret


def generate_cnot_ctrl_templates() -> List[OptimizationTemplate]:
    tpl_list = [
        [2, 0, [[CX, [0, 1]]]],
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
