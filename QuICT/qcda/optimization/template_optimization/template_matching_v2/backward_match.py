from typing import List, Tuple
from collections import deque
import heapq

from .matching_dag_circuit import MatchingDAGCircuit, MatchingDAGNode, NodeInfo


class MatchingScenario:
    def __init__(self, circuit_info, template_info, match, counter):
        self.circuit_info = circuit_info
        self.template_info = template_info
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
        # DONE why starting from t_node_id?
        # DONE why not successors
        successors = set(template.all_successors(t_node_id))
        matches = {m[0] for m in match}
        ret = sorted(filter(
            lambda x: x not in successors and x not in matches and not template_info[x].is_blocked,
            range(t_node_id + 1, template.size)
        ), reverse=True)

        return [template.get_node(i) for i in ret]

    @classmethod
    def _prune(cls, scenarios: deque, gate_indices, depth, width):
        counters = [s.counter for s in scenarios]
        if counters.count(max(counters)) == len(counters) and counters[0] <= len(gate_indices) \
                and counters[0] % depth == 0:
            ret = heapq.nlargest(width, scenarios, key=lambda s: len(s.match))
            return ret
        return scenarios

    @classmethod
    def _left_block(cls,
                    circuit: MatchingDAGCircuit,
                    template: MatchingDAGCircuit,
                    scenario: MatchingScenario,
                    c_node_id: int):
        c_info = scenario.circuit_info.copy()
        t_info = scenario.template_info.copy()
        match = scenario.match.copy()
        counter = scenario.counter

        c_info[c_node_id] = NodeInfo(None, True)
        for c_nxt_id in circuit.all_predecessors(c_node_id):
            c_info[c_nxt_id] = NodeInfo(None, True)

        nxt_scenario = MatchingScenario(c_info, t_info, match, counter+1)
        return nxt_scenario

    @classmethod
    def _right_block(cls,
                     circuit: MatchingDAGCircuit,
                     template: MatchingDAGCircuit,
                     scenario: MatchingScenario,
                     c_node_id: int):

        c_info = scenario.circuit_info.copy()
        t_info = scenario.template_info.copy()
        match = scenario.match
        counter = scenario.counter

        c_info[c_node_id] = NodeInfo(None, True)

        broken_match = []
        for c_nxt_id in circuit.all_successors(c_node_id):
            t_nxt_id = c_info[c_nxt_id].matched_with
            if t_nxt_id is not None:
                broken_match.append(t_nxt_id)
                t_info[t_nxt_id] = NodeInfo(None, False)
            c_info[c_nxt_id] = NodeInfo(None, True)

        new_match = list(filter(lambda x: x[0] not in broken_match, match))
        nxt_scenario = MatchingScenario(c_info, t_info, new_match, counter + 1)

        return broken_match, nxt_scenario

    @classmethod
    def execute(cls,
                circuit: MatchingDAGCircuit,
                template: MatchingDAGCircuit,
                forward_match: List[Tuple[int, int]],
                c_node_id: int,
                t_node_id: int,
                qubit_mapping: List[int],
                prune_param=None) -> List[List[Tuple[int, int]]]:

        gate_indices = cls._calc_gate_indices(circuit)
        scenarios = deque([
            MatchingScenario(
                circuit.matching_info(),
                template.matching_info(),
                forward_match,
                0
            )
        ])

        # FIXME (t_node_id - 1) or t_node_id
        remain_cnt = template.size - t_node_id - len(forward_match)

        res = []
        while len(scenarios) > 0:
            if prune_param is not None:
                scenarios = cls._prune(scenarios, gate_indices, *prune_param)

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
            if c_info[cur_c_node_id].is_blocked:
                nxt_scenario = MatchingScenario(c_info, t_info, match, counter + 1)
                # FIXME stack or queue
                scenarios.append(nxt_scenario)
                continue

            cands = cls._find_candidates(match, template, t_info, t_node_id)
            flag_broken = True  # if all possible matches will break sth
            flag_succeed = False  # if there is possible match

            for cur_t_node in cands:
                # FIXME remove identical matches

                if cur_t_node.compare_with(cur_c_node, qubit_mapping):
                    cur_c_info = c_info.copy()
                    cur_t_info = t_info.copy()
                    cur_match = match.copy()

                    # block_list = []
                    broken_match = []

                    for block_id in template.all_successors(cur_t_node.id):
                        if cur_t_info[block_id].matched_with is None:
                            cur_t_info[block_id] = NodeInfo(None, True)

                            # FIXME do we need block_list
                            for t_nxt_id in template.all_successors(block_id):
                                c_nxt_id = cur_t_info[t_nxt_id].matched_with
                                if c_nxt_id is not None:
                                    # FIXME do we need to block c_nxt_id
                                    cur_c_info[c_nxt_id] = NodeInfo(None, True)
                                    broken_match.append(t_nxt_id)
                                cur_t_info[t_nxt_id] = NodeInfo(None, True)

                    flag_broken &= len(broken_match) > 0
                    new_match = list(filter(lambda x: x[0] not in broken_match, cur_match))

                    # check if fixed match is unchanged
                    if (t_node_id, c_node_id) in new_match and \
                            all(map(lambda x: x in new_match, backward_match)):
                        cur_t_info[cur_t_node.id] = NodeInfo(cur_c_node_id, False)
                        cur_c_info[cur_c_node_id] = NodeInfo(cur_t_node.id, False)
                        new_match.append((cur_t_node.id, cur_c_node_id))
                        nxt_scenario = MatchingScenario(cur_c_info, cur_t_info, new_match, counter + 1)
                        # print('!', new_match)
                        scenarios.append(nxt_scenario)

                        flag_succeed = True

            if flag_succeed:
                broken_match, nxt_scenario = cls._right_block(circuit, template, cur_scenario, cur_c_node_id)

                if (t_node_id, c_node_id) in nxt_scenario.match and \
                        all(map(lambda x: x in nxt_scenario.match, backward_match)):
                    scenarios.append(nxt_scenario)

                if broken_match and flag_broken:
                    nxt_scenario = cls._left_block(circuit, template, cur_scenario, cur_c_node_id)
                    scenarios.append(nxt_scenario)
            else:
                c_info[cur_c_node_id] = NodeInfo(None, True)
                following_match = list(filter(
                    lambda i: c_info[i].matched_with is not None,
                    circuit.all_successors(cur_c_node_id)
                ))
                predecessors = circuit.all_predecessors(cur_c_node_id)
                if not following_match or not predecessors:
                    nxt_scenario = MatchingScenario(c_info, t_info, match, counter + 1)
                    scenarios.append(nxt_scenario)
                else:
                    nxt_scenario = cls._left_block(circuit, template, cur_scenario, cur_c_node_id)
                    scenarios.append(nxt_scenario)

                    broken_match, nxt_scenario = cls._right_block(circuit, template, cur_scenario, cur_c_node_id)

                    if (t_node_id, c_node_id) in nxt_scenario.match and \
                            all(map(lambda x: x in nxt_scenario.match, backward_match)):
                        scenarios.append(nxt_scenario)

        max_len = max(len(match) for match in res)
        maximal_res = filter(lambda m: len(m) == max_len, res)
        unique_res = set(map(tuple, maximal_res))
        return list(map(list, unique_res))
