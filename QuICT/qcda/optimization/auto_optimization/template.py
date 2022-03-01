from typing import Dict, Tuple
from collections import deque

from QuICT.core import *
from .dag import DAG


class OptimizationTemplate:
    def __init__(self, template: DAG, replacement: DAG = None, anchor: int = 0):
        self.template = template
        self.replacement = replacement
        self.anchor = anchor

    def replace(self, mapping: Dict[int, DAG.Node]):
        assert self.replacement, "Template has no replacement"

        # TODO code review, try merging?
        replacement = self.replacement.copy()
        for qubit_ in range(self.template.size):
            # first node on qubit_ in template circuit 
            t_node, t_qubit = self.template.start_nodes[qubit_].successors[0]
            # first node on qubit_ in replacement circuit
            r_node, r_qubit = replacement.start_nodes[qubit_].successors[0]
            # node previous to the node corresponding to t_node in original circuit
            p_node, p_qubit = mapping[id(t_node)].predecessors[t_qubit]
            # place r_node after p_node
            p_node.connect(p_qubit, r_qubit, r_node)

        for qubit_ in range(self.template.size):
            # last node on qubit_ in template circuit
            t_node, t_qubit = self.template.end_nodes[qubit_].predecessors[0]
            # last node on qubit_ in replacement circuit
            r_node, r_qubit = replacement.end_nodes[qubit_].predecessors[0]
            # node successive to the node corresponding to t_node in original circuit
            s_node, s_qubit = mapping[id(t_node)].successors[t_qubit]
            # place s_node after r_node
            r_node.connect(r_qubit, s_qubit, s_node)

        for each in mapping.values():
            del each

    def compare(self, other: Tuple[DAG.Node, int]):
        o_node, o_qubit = other
        t_node, t_qubit = self.template.start_nodes[self.anchor].successors[0]
        if o_node.gate.qasm_name != t_node.gate.qasm_name or o_qubit != t_qubit or o_node.flag:
            return None

        mapping = {id(t_node): o_node}
        queue = deque([(t_node, o_node)])
        while len(queue) > 0:
            u, v = queue.popleft()
            for neighbors in ['predecessors', 'successors']:
                for qubit_ in range(u.size):
                    u_nxt, u_qubit = getattr(u, neighbors)[qubit_]
                    assert u_nxt, "u_nxt == None should not happen"
                    if not u_nxt.gate or id(u_nxt) in mapping:
                        continue

                    v_nxt, v_qubit = getattr(v, neighbors)[qubit_]
                    assert v_nxt, "v_nxt == None should not happen"
                    if not v_nxt.gate or u_qubit != v_qubit or v_nxt.flag or \
                            u_nxt.gate.qasm_name != v_nxt.gate.qasm_name:
                        return None

                    mapping[id(u_nxt)] = v_nxt
                    queue.append((id(u_nxt), v_nxt))

        for each in mapping.values():
            each.flag = 1
        return mapping

    def replace_all(self, dag: DAG):
        for node in dag.topological_sort():
            node.flag = 0

        matched = []
        for node in dag.topological_sort():
            for qubit_ in range(node.size):
                mapping = self.compare((node, qubit_))
                if not mapping:
                    continue
                matched.append(mapping)

        for mapping in matched:
            self.replace(mapping)

        return len(matched)
