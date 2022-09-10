import itertools
from typing import List

from QuICT.core import Circuit
from .matching_dag_circuit import MatchingDAGCircuit, MatchingDAGNode
from .backward_match import BackwardMatch
from .forward_match import ForwardMatch


class TemplateMatching:
    """
    [1] Iten, R., Moyard, R., Metger, T., Sutter, D. and Woerner, S., 2020.
    Exact and practical pattern matching for quantum circuit optimization.
    `arXiv:1909.05270 <https://arxiv.org/abs/1909.05270>`_
    """

    @classmethod
    def _qubit_fixing(cls,
                      circuit: MatchingDAGCircuit,
                      template: MatchingDAGCircuit,
                      c_node_id: int,
                      t_node_id: int,
                      cnt: int):

        t_successors = template.all_successors(t_node_id)
        c_successors = circuit.all_successors(c_node_id)

        # FIXME review this condition
        cands = c_successors if 2 * len(t_successors) > template.size - t_node_id - 1 \
            else sorted(set(range(circuit.size)) - set(c_successors), reverse=True)

        ret = set(circuit.get_node(c_node_id).qargs)
        for cand in cands[: cnt]:
            cur = ret | set(circuit.get_node(cand).qargs)
            if len(cur) > cnt:
                break
            ret = cur
        return list(ret)

    @classmethod
    def execute(cls,
                circuit: MatchingDAGCircuit,
                template: MatchingDAGCircuit,
                qubit_fixing_param: List[int] = None,
                prune_param: List[int] = None
                ):

        ret = []
        for t_node_id in range(template.size):
            for c_node_id in range(circuit.size):
                t_node = template.get_node(t_node_id)
                c_node = circuit.get_node(c_node_id)
                if not t_node.compare_with(c_node):
                    continue

                fixed_q = [] if qubit_fixing_param is None \
                    else cls._qubit_fixing(circuit, template, c_node_id, t_node_id, *qubit_fixing_param)

                all_free_q = set(range(circuit.size)) - set(fixed_q)
                for free_q in itertools.combinations(all_free_q, template.size - len(fixed_q)):
                    q_set = list(free_q) + fixed_q
                    for mapping in itertools.permutations(q_set):
                        if not t_node.compare_with(c_node, mapping):
                            continue

                        forward_match = ForwardMatch.execute(
                            circuit, template, c_node_id, t_node_id, list(mapping))
                        match_list = BackwardMatch.execute(
                            circuit, template, forward_match, c_node_id, t_node_id, list(mapping), prune_param)

                        # TODO do sth
                        ret.extend(match_list)

                # for qubit_mapping in itertools.permutations(range(circuit.size), template.size):
                #     if not t_node.compare_with(c_node, qubit_mapping):
                #         continue
                #
                #     forward_match = ForwardMatch.execute(
                #         circuit, template, c_node_id, t_node_id, list(qubit_mapping))
                #     match_list = BackwardMatch.execute(
                #         circuit, template, forward_match, c_node_id, t_node_id, list(qubit_mapping), prune_param)
                #     ret.extend(match_list)
        return ret
