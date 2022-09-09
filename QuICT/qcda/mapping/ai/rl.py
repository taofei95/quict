import math
from copy import copy, deepcopy
from random import choice, random
from typing import List, Set, Tuple, Union

import torch
from QuICT.qcda.mapping.ai.data_factory import DataFactory
from QuICT.qcda.mapping.ai.gnn_mapping import GnnMapping


class State:
    def __init__(
        self,
        layered_circ: List[Set[Tuple[int, int]]],
        topo_name: str,
        x: torch.IntTensor,
        edge_index: torch.IntTensor,
        cur_mapping: List[int],
    ) -> None:
        self.layered_circ = layered_circ
        self.topo_name = topo_name
        self.x = x
        self.edge_index = edge_index
        self.cur_mapping = cur_mapping


class Transition:
    def __init__(
        self, state: State, action: Tuple[int, int], next_state: State, reward: float
    ) -> None:
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward

    def __iter__(self):
        return iter((self.state, self.action, self.next_state, self.reward))


class Agent:
    def __init__(
        self,
        max_qubit_num: int = 30,
        max_layer_num: int = 60,
        inner_feat_dim: int = 50,
        epsilon_start: float = 0.9,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 100.0,
    ) -> None:
        # Copy values in.
        self._max_qubit_num = max_qubit_num
        self._max_layer_num = max_layer_num
        self._feat_dim = inner_feat_dim
        self._epsilon_start = epsilon_start
        self._epsilon_end = epsilon_end
        self._epsilon_decay = epsilon_decay

        # Exploration related.
        self._explore_step = 0
        self._state = None  # Reset during training

        # Random data generator
        self._data_factory = DataFactory(
            max_qubit_num=max_qubit_num,
            max_layer_num=max_layer_num,
        )

    def reset_explore_state(self):
        self._state = State(*self._data_factory.get_one())
        self._explore_step = 0

    def select_action(
        self,
        policy_net: GnnMapping,
        policy_net_device: str,
    ) -> Tuple[int, int]:
        eps_threshold = self._epsilon_end + (
            self._epsilon_start - self._epsilon_end
        ) * math.exp(-1.0 * self._explore_step / self._epsilon_decay)
        self._explore_step += 1
        sample = random()
        edges = self._data_factory.topo_edge_map[self._state.topo_name]
        if sample > eps_threshold:
            max_q = self._max_qubit_num
            q = self._data_factory.topo_qubit_num_map[self._state.topo_name]

            # Chose an action based on policy_net
            x = self._state.x.to(policy_net_device)
            edge_index = self._state.edge_index.to(policy_net_device)
            attn = policy_net(x, edge_index).detach().cpu()
            attn = attn.view(max_q, max_q)

            # Use a mask matrix to filter out unwanted qubit pairs.
            mask = self._data_factory.topo_mask_map[self._state.topo_name]
            attn = attn[:q, :q] * mask
            pos = int(torch.argmax(attn))
            del attn
            u, v = pos // q, pos % q
            return u, v
        else:
            return choice(edges)

    def _take_action(
        self, action: Tuple[int, int]
    ) -> Tuple[Union[State, None], float, bool]:
        """Take given action on trainer's state.

        Args:
            action (Tuple[int, int]): Swap gate used on current topology.

        Returns:
            Tuple[Union[State, None], bool]: Tuple of (Next State, Terminated).
        """
        u, v = action
        graph = self._data_factory.topo_graph_map[self._state.topo_name]
        if not graph.has_edge(u, v):
            reward = 0.0
            next_state = self._state
            return next_state, reward, False

        next_mapping = deepcopy(self._state.cur_mapping)
        next_mapping[u], next_mapping[v] = next_mapping[v], next_mapping[u]
        next_layered_circ = self._data_factory.remap_layered_circ(
            self._state.layered_circ, next_mapping
        )
        topo_dist = self._data_factory.topo_dist_map[self._state.topo_name]
        remove_layer_num = 0
        reward = 0.0
        for layer in next_layered_circ:
            for x, y in copy(layer):
                if topo_dist[x][y] == 1:
                    layer.remove((x, y))
                    reward += 1.0
            if len(layer) > 0:
                break
            remove_layer_num += 1
        next_layered_circ = next_layered_circ[remove_layer_num:]

        # Check if there are only padded empty layers left.
        terminated = len(next_layered_circ) == 0
        if terminated:
            return None, reward, True

        # X for the same topology is always the same, so there's ne need to copy.
        next_x = self._state.x

        # Generate new edge_index
        topo_graph = self._data_factory.topo_graph_map[self._state.topo_name]
        next_edge_index = self._data_factory.get_circ_edge_index(
            layered_circ=next_layered_circ,
            topo_graph=topo_graph,
        )

        next_state = State(
            layered_circ=next_layered_circ,
            topo_name=self._state.topo_name,
            x=next_x,
            edge_index=next_edge_index,
            cur_mapping=next_mapping,
        )

        return next_state, reward, False
