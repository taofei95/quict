from typing import List, Tuple

from .matching_dag_circuit import MatchingDAGCircuit, MatchingDAGNode


class ForwardMatch:

    @classmethod
    def _insert_match_nodes(cls, match_nodes: List[MatchingDAGNode], node: MatchingDAGNode):
        # TODO code review
        key = node.successors_to_visit
        lb, rb = 0, len(match_nodes) - 1
        # bin search first element that > key
        while lb < rb:
            mid = (lb + rb) // 2
            if match_nodes[mid].successors_to_visit <= key:
                lb = mid + 1
            else:
                # FIXME mid or mid-1
                rb = mid
        match_nodes.insert(lb - 1, node)

    @classmethod
    def _find_candidates(cls,
                         match: List[Tuple[int, int]],
                         template: MatchingDAGCircuit,
                         t_node_id: int) -> List[MatchingDAGNode]:
        # TODO code review
        match = {match[i][0] for i in range(len(match))}
        # block = reduce(set.__or__, [template.all_successors(i) for i in (match - {t_node_id})])
        block = template.all_successors((match - {t_node_id}))
        return template.get_node(t_node_id).successors - match - block

    @classmethod
    def execute(cls,
                circuit: MatchingDAGCircuit,
                template: MatchingDAGCircuit,
                c_node_id: int,
                t_node_id: int,
                qubit_mapping: List[int]):

        circuit.init_forward_matching(c_node_id, t_node_id, s2v_enabled=True)
        template.init_forward_matching(t_node_id, c_node_id)

        match = [(t_node_id, c_node_id)]
        match_nodes = [circuit.get_node(c_node_id)]
        while len(match_nodes) > 0:
            cur_node: MatchingDAGNode = match_nodes[0]
            match_nodes.pop(0)

            # if not cur_node.successors_to_visit:
            #     # FIXME: can we remove this if
            #     continue

            # TODO be careful to return node object
            nxt_node: MatchingDAGNode = cur_node.pop_successors_to_visit()
            if nxt_node is None:
                continue

            if cur_node.successors_to_visit:
                cls._insert_match_nodes(match_nodes, cur_node)

            if not cur_node.matchable():
                continue

            cands: List[MatchingDAGNode] = cls._find_candidates(match, template, cur_node.matched_with)
            success = False
            for t_nxt_node in cands:
                if t_nxt_node.compare_with(nxt_node, qubit_mapping):
                    t_nxt_node.matched_with = nxt_node.id
                    nxt_node.matched_with = t_nxt_node.id
                    match.append([t_nxt_node.id, nxt_node.id])

                    # calc nxt_node.successors_to_visit
                    nxt_node.successors_to_visit = sorted(list(filter(
                        lambda x: circuit.get_node(x).matchable,
                        nxt_node.successors
                    )))
                    match_nodes.append(nxt_node)

                    success = True
                    break

            if not success:
                nxt_node.is_blocked = True
                for node_id in circuit.all_successors(nxt_node.id):
                    succ_node = circuit.get_node(node_id)

                    succ_node.is_blocked = True
                    # remove matched info
                    if succ_node.matched_with:
                        t = succ_node.matched_with
                        template.get_node(t).matched_with = None
                        succ_node.matched_with = None
                        match.remove([t, succ_node.id])

        return match
