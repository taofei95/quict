from typing import List, Tuple
from collections import deque
import heapq


from .matching_dag_circuit import MatchingDAGCircuit, MatchingDAGNode


class MatchingScenario:
    def __init__(self, circuit_info, template_info, match, counter):
        # FIXME copy needed?
        self.circuit_info = circuit_info.copy()
        self.template_info = template_info.copy()
        self.match = match.copy()
        self.counter = counter


class BackwardMatch:
    @classmethod
    def _calc_gate_indices(cls, circuit: MatchingDAGCircuit):
        return list(filter(
            lambda i: circuit.get_node(i).matchable(),
            reversed(range(circuit.size))
        ))

    @classmethod
    def _find_candidates(cls, match, template, template_info, t_node_id) -> List[MatchingDAGNode]:
        # FIXME why starting from t_node_id?
        # FIXME why not successors
        successors = set(template.all_successors(t_node_id))
        matches = {m[0] for m in match}
        ret = sorted(filter(
            lambda x: x not in successors and x not in matches and not template_info[x][1],
            range(template.size)
        ), reverse=True)

        return [template.get_node(i) for i in ret]

    @classmethod
    def _prune(cls, scenarios: deque[MatchingScenario], gate_indices, depth, width):
        counters = [s.counter for s in scenarios]
        if counters.count(max(counters)) == len(counters) and counters[0] <= len(gate_indices) \
                and counters[0] % depth == 0:
            ret = heapq.nlargest(width, scenarios, key=lambda s: len(s.match))
            return ret
        return scenarios

    @classmethod
    def execute(cls,
                circuit: MatchingDAGCircuit,
                template: MatchingDAGCircuit,
                forward_match: List[Tuple[int, int]],
                c_node_id: int,
                t_node_id: int,
                qubit_mapping: List[int],
                prune_param=None) -> List[List[Tuple[int, int]]]:
        # TODO add heuristics_backward_param

        gate_indices = cls._calc_gate_indices(circuit)
        scenarios = deque([
            MatchingScenario(
                circuit.matching_info(),
                template.matching_info(),
                forward_match,
                0
            )
        ])

        # FIXME why - (t_node_id - 1)
        remain_cnt = template.size - (t_node_id - 1) - len(forward_match)

        res = []
        while len(scenarios) > 0:
            if prune_param is not None:
                scenarios = cls._prune(scenarios, gate_indices, *prune_param)

            # TODO add heuristics here

            cur_scenario = scenarios.pop()
            c_info = cur_scenario.circuit_info
            t_info = cur_scenario.template_info
            match = cur_scenario.match
            counter = cur_scenario.counter
            backward_match = list(filter(lambda x: x not in forward_match, match))

            if counter >= len(gate_indices) or len(backward_match) == remain_cnt:
                res.append(sorted(match))
                continue

            cur_c_node_id = gate_indices[counter]
            cur_c_node = circuit.get_node(cur_c_node_id)
            if c_info[cur_c_node_id][1]:
                nxt_scenario = MatchingScenario(c_info, t_info, match, counter + 1)
                scenarios.appendleft(nxt_scenario)

            cands = cls._find_candidates(match, template, t_info, t_node_id)
            flag_broken = True   # if all possible matches will break sth
            flag_succeed = False # if there is possible match

            for cur_t_node in cands:
                # FIXME remove identical matches

                if cur_t_node.compare_with(cur_c_node, qubit_mapping):
                    cur_c_info = c_info.copy()
                    cur_t_info = t_info.copy()
                    cur_match = match.copy()

                    # block_list = []
                    broken_match = []

                    for block_id in template.all_successors(cur_t_node.id):
                        if cur_t_info[block_id][0] is None:
                            cur_t_info[block_id] = (None, True)
                            # block_list.append(potential_block)

                            # FIXME do we need block_list
                            for t_nxt_id in template.all_successors(block_id):
                                c_nxt_id = cur_t_info[t_nxt_id][0]
                                if c_nxt_id is not None:
                                    # FIXME do we need to block c_nxt_id
                                    cur_c_info[c_nxt_id] = (None, False)
                                    broken_match.append(t_nxt_id)
                                cur_t_info[t_nxt_id] = (None, True)

                    flag_broken &= len(broken_match) > 0
                    new_match = list(filter(lambda x: x[0] not in broken_match, cur_match))

                    # check if fixed match is unchanged
                    if (t_node_id, c_node_id) in new_match and \
                            all(map(lambda x: x in new_match, backward_match)):

                        cur_t_info[cur_t_node.id] = (cur_c_node_id, False)
                        cur_c_info[cur_c_node_id] = (cur_t_node.id, False)
                        new_match.append((cur_t_node.id, cur_c_node_id))
                        nxt_scenario = MatchingScenario(cur_c_info, cur_t_info, new_match, counter + 1)
                        scenarios.append(nxt_scenario)

                        flag_succeed = True

            if flag_succeed:
                cur_c_info = c_info.copy()
                cur_t_info = t_info.copy()
                cur_match = match.copy()
                cur_c_info[cur_c_node_id] = (None, True)

                broken_match = []
                for c_nxt_id in circuit.all_successors(cur_c_node_id):
                    t_nxt_id = cur_c_info[c_nxt_id][0]
                    if t_nxt_id is not None:
                        broken_match.append(c_nxt_id)
                        cur_t_info[t_nxt_id] = (None, True)
                    cur_c_info[c_nxt_id] = (None, True)

                new_match = list(filter(lambda x: x[0] not in broken_match, cur_match))
                if (t_node_id, c_node_id) in new_match and \
                        all(map(lambda x: x in new_match, backward_match)):
                    nxt_scenario = MatchingScenario(cur_c_info, cur_t_info, new_match, counter+1)
                    scenarios.append(nxt_scenario)

                if broken_match and flag_broken:
                    cur_c_info = c_info.copy()
                    cur_t_info = t_info.copy()
                    cur_match = match.copy()
                    cur_c_info[cur_c_node_id] = (None, True)

                    for c_nxt_id in circuit.all_predecessors(cur_c_node_id):
                        cur_c_info[c_nxt_id] = (None, True)

                    nxt_scenario = MatchingScenario(cur_c_info, cur_t_info, match, counter+1)
                    scenarios.append(nxt_scenario)
            else:
                c_info[cur_c_node_id] = (None, True)
                following_match = list(filter(
                    lambda i: c_info[i][0] is not None,
                    circuit.all_successors(cur_c_node_id)
                ))
                predecessors = circuit.all_predecessors(cur_c_node_id)
                if not following_match or not predecessors:
                    nxt_scenario = MatchingScenario(c_info, t_info, match, counter+1)
                    scenarios.append(nxt_scenario)
                else:
                    # block predecessors
                    cur_c_info = c_info.copy()
                    cur_t_info = t_info.copy()
                    cur_match = match.copy()

                    for c_nxt_id in predecessors:
                        cur_c_info[c_nxt_id] = (None, True)
                    nxt_scenario = MatchingScenario(cur_c_info, cur_t_info, cur_match, counter+1)
                    scenarios.append(nxt_scenario)

                    # block successors
                    cur_c_info = c_info.copy()
                    cur_t_info = t_info.copy()
                    cur_match = match.copy()

                    broken_match = []
                    for c_nxt_id in circuit.all_successors(cur_c_node_id):
                        t_nxt_id = c_info[c_nxt_id][0]
                        if t_nxt_id is not None:
                            broken_match.append(c_nxt_id)
                            # FIXME not needed?
                            cur_t_info[t_nxt_id] = (None, True)
                        c_info[c_nxt_id] = (None, True)
                    new_match = list(filter(lambda x: x[0] not in broken_match, cur_match))

                    if (t_node_id, c_node_id) in new_match and \
                            all(map(lambda x: x in new_match, backward_match)):
                        nxt_scenario = MatchingScenario(cur_c_info, cur_t_info, new_match, counter+1)
                        scenarios.append(nxt_scenario)

        max_len = max(len(match) for match in res)
        maximal_res = filter(lambda m: len(m) == max_len, res)
        unique_res = set(map(tuple, maximal_res))
        return list(map(list, unique_res))
