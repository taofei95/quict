import math
import os
import os.path as osp
from collections import deque, namedtuple
from random import choice, randint, random
from typing import List, Set, Tuple, Union
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from QuICT.qcda.mapping.ai.gtdqn import GraphTransformerDeepQNetwork


class State:
    def __init__(
        self,
        layered_circ: List[Set[Tuple[int, int]]],
        topo_name: str,
        x: torch.IntTensor,
        spacial_encoding: torch.IntTensor,
        cur_mapping: List[int],
    ) -> None:
        self.layered_circ = layered_circ
        self.topo_name = topo_name
        self.x = x
        self.spacial_encoding = spacial_encoding
        self.cur_mapping = cur_mapping


class Transition:
    def __init__(
        self, state: State, action: Tuple[int, int], next_state: State, reward: float
    ) -> None:
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward


class Trainer:
    def __init__(
        self,
        batch_size: int = 16,
        total_epoch: int = 200,
        explore_period: int = 100,
        target_update_period: int = 100,
        epsilon_start: float = 0.9,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 100.0,
        device: str = "cpu",
        model_path: str = None,
        log_dir: str = None,
    ) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self._device = device

        # self._dataset = MappingGraphormerDataset()
        # max_qubit_number, max_layer_num = self._dataset.metadata

        self._explore_step = 0
        self._state = None
        self._reset_explore()

        # TODO: parameterize this model.
        self._policy_net = GraphTransformerDeepQNetwork(
            max_qubit_num=max_qubit_number,
            max_layer_num=max_layer_num,
            inner_feat_dim=30,
            head=3,
        )
        self._target_net = GraphTransformerDeepQNetwork(
            max_qubit_num=max_qubit_number,
            max_layer_num=max_layer_num,
            inner_feat_dim=30,
            head=3,
        )
        self._target_net.load_state_dict(self._policy_net.state_dict())

        if model_path is None:
            model_path = osp.dirname(osp.abspath(__file__))
            model_path = osp.join(model_path, "model_graphormer")
        if not osp.exists(model_path):
            os.makedirs(model_path)
        self._model_path = model_path

        self._batch_size = batch_size
        self._total_epoch = total_epoch
        self._explore_period = explore_period
        self._target_update_period = target_update_period
        self._epsilon_start = epsilon_start
        self._epsilon_end = epsilon_end
        self._epsilon_decay = epsilon_decay

        self._loss = nn.SmoothL1Loss()

        if log_dir is None:
            log_dir = osp.dirname(osp.abspath(__file__))
            log_dir = osp.join(log_dir, "torch_runs")
            log_dir = osp.join(log_dir, "circuit_graphormer")

        self._writer = SummaryWriter(log_dir=log_dir)

        print(f"Start training on {device}...")

    def _reset_explore(
        self,
    ):
        self._state = State(*choice(self._dataset))

    def _select_action(
        self,
    ) -> Tuple[int, int]:
        eps_threshold = self._epsilon_end + (
            self._epsilon_start - self._epsilon_end
        ) * math.exp(-1.0 * self._explore_step / self._epsilon_decay)
        self._explore_step += 1
        sample = random()
        graph = self._dataset._topo_graph_map[self._state.topo_name]
        edges = self._dataset._topo_edge_map[[self._state.topo_name]]
        if sample > eps_threshold:
            # Chose an action based on policy_net
            q_mat = self._policy_net(self._state.x, self._state.spacial_encoding)
            topo_qubit_num = self._dataset._topo_qubit_num_map[self._state.topo_name]
            q_mat = (
                q_mat[:topo_qubit_num, :topo_qubit_num]
                * self._dataset._topo_mask_map[self._state.topo_name]
            )
            pos = torch.argmax(q_mat)
            u, v = pos // topo_qubit_num, pos % topo_qubit_num
            if graph.has_edge(u, v):
                # Policy net gives a correct output. Return it.
                action = u, v
            else:
                # Policy net gives wrong output. Now we gives a random output
                action = choice(edges)
            return action
        else:
            return choice(edges)

    def _take_action(self, action: Tuple[int, int]) -> Tuple[Union[State, None], bool]:
        """Take given action on trainer's state.

        Args:
            action (Tuple[int, int]): Swap gate used on current topology.

        Returns:
            Tuple[Union[State, None], bool]: Tuple of (Next State, finished).
        """
        next_layered_circ = deepcopy(self._state.layered_circ)
        u, v = action
        next_layered_circ[0].remove((u, v))
        next_layered_circ[0].remove((v, u))
        if len(next_layered_circ[0]) == 0:
            next_layered_circ.pop(0)
        # Check if there are only padded empty layers left.
        flag = True
        for layer in next_layered_circ:
            flag = len(layer) == 0
        if flag:
            return None, True

        next_mapping = deepcopy(self._state.cur_mapping)
        next_mapping[u], next_mapping[v] = next_mapping[v], next_mapping[u]
        next_layered_circ = self._dataset.processor.remap_layered_circ(
            next_layered_circ, next_mapping
        )
        next_x = self._state.x
        next_circ_graph = self._dataset.processor.get_circ_graph(
            layered_circ=next_layered_circ
        )
        next_topo_graph = self._dataset._topo_graph_map[self._state.topo_name]
        next_spacial_encoding = self._dataset.processor.get_spacial_encoding(
            circ_graph=next_circ_graph,
            topo_graph=next_topo_graph,
            cur_mapping=next_mapping,
        )

        next_state = State(
            layered_circ=next_layered_circ,
            topo_name=self._state.topo_name,
            x=next_x,
            spacial_encoding=next_spacial_encoding,
            cur_mapping=next_mapping,
        )

        return next_state, False

    def train_one_epoch(self, epoch: int):
        pass
