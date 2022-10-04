from itertools import chain
from typing import List
from functools import cmp_to_key

from QuICT.core import Circuit
from .matching_dag_circuit import MatchingDAGCircuit, MatchingDAGNode, Match


GATE_COST = {
    'id': 0, 'x': 1, 'y': 1, 'z': 1, 'h': 1, 't': 1, 'tdg': 1, 's': 1, 'sdg': 1,
    'u1': 1, 'u2': 2, 'u3': 2, 'rx': 1, 'ry': 1, 'rz': 1, 'r': 2, 'cx': 2,
    'cy': 4, 'cz': 4, 'ch': 8, 'swap': 6, 'iswap': 8, 'rxx': 9, 'ryy': 9,
    'rzz': 5, 'rzx': 7, 'ms': 9, 'cu3': 10, 'crx': 10, 'cry': 10, 'crz': 10,
    'ccx': 21, 'rccx': 12, 'c3x': 96, 'rc3x': 24, 'c4x': 312, 'p': 1
}


class Substitution:
    def __init__(self,
                 circuit: MatchingDAGCircuit,
                 template: MatchingDAGCircuit,
                 match: Match):
        self.circuit = circuit
        self.template = template
        self.match = match
        self._cost = None
        self._pred_set = None
        self._pred = None

    @property
    def cost(self):
        if self._cost is None:
            old_nodes = self.match.template_nodes
            new_nodes = set(range(self.template.size)) - old_nodes
            old_cost = sum(GATE_COST[self.template.get_node(n).name] for n in old_nodes)
            new_cost = sum(GATE_COST[self.template.get_node(n).name] for n in new_nodes)
            self._cost = old_cost - new_cost
        return self._cost

    @property
    def pred(self):
        if self._pred is None:
            self._pred = sorted(self.pred_set)
        return self._pred

    @property
    def pred_set(self):
        if self._pred_set is None:
            self._pred_set = self.circuit.all_predecessors(self.match.circuit_nodes)
        return self._pred_set

    def cmp_with(self, other):
        assert isinstance(other, Substitution)
        return -1 if self.match.circuit_nodes & other.pred_set else 1

    def get_substitution(self):
        pred = self.template.all_predecessors(self.match.template_nodes)
        succ = set(self.template.nodes()) - self.match.template_nodes - pred

        circ = Circuit(self.circuit.width)
        for node_id in chain(sorted(pred, reverse=True), sorted(succ, reverse=True)):
            node: MatchingDAGNode = self.template.get_node(node_id)
            qubits = list([self.match.qubit_mapping[i] for i in node.qargs])
            node.gate.inverse().copy() | circ(qubits)
        return circ


class TemplateSubstitution:

    @classmethod
    def _calc_sub_list(cls,
                       circuit: MatchingDAGCircuit,
                       template: MatchingDAGCircuit,
                       match_list: List[Match]):
        all_subs = [Substitution(circuit, template, m) for m in match_list]

        sub_list = []
        visited_nodes = set()
        for sub in sorted(filter(lambda s: s.cost > 0, all_subs),
                          key=lambda s: s.cost,
                          reverse=True):
            cur_nodes = sub.match.circuit_nodes
            if visited_nodes & cur_nodes:
                continue

            flag_intersect = False
            for other_sub in sub_list:
                if (other_sub.pred_set & cur_nodes) and \
                        (sub.pred_set & other_sub.match.circuit_nodes):
                    flag_intersect = True
                    break

            if not flag_intersect:
                visited_nodes |= cur_nodes
                sub_list.append(sub)

        return sorted(sub_list, key=cmp_to_key(Substitution.cmp_with))

    @classmethod
    def execute(cls,
                circuit: MatchingDAGCircuit,
                template: MatchingDAGCircuit,
                match_list: List[Match]):

        sub_list = cls._calc_sub_list(circuit, template, match_list)

        new_circ = Circuit(circuit.width)
        visited_nodes = set()

        for sub in sub_list:
            for node_id in filter(lambda x: x not in visited_nodes, sub.pred):
                node: MatchingDAGNode = circuit.get_node(node_id)
                new_circ.append(node.gate.copy())

            new_circ.extend(sub.get_substitution().gates)

            visited_nodes.update(sub.pred)
            visited_nodes.update(sub.match.circuit_nodes)

        for node_id in filter(lambda x: x not in visited_nodes, circuit.nodes()):
            node: MatchingDAGNode = circuit.get_node(node_id)
            new_circ.append(node.gate.copy())

        return MatchingDAGCircuit(new_circ)
