import copy
import math
from random import choice, random
from typing import Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
import torch
from QuICT.core import *
from QuICT.core.gate import *
from QuICT.core.utils import CircuitBased
from QuICT.qcda.mapping.ai.data_def import (
    CircuitInfo,
    DataFactory,
    State,
    Transition,
)
from QuICT.qcda.mapping.ai.nn_mapping import NnMapping
from torch_geometric.data import Batch as PygBatch


class Agent:
    def __init__(
        self,
        topo: Union[Layout, str],
        max_gate_num: int,
        epsilon_start: float = 0.9,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 100.0,
        reward_scale: float = 5.0,
    ) -> None:
        # Copy values in.
        self._max_gate_num = max_gate_num
        self._epsilon_start = epsilon_start
        self._epsilon_end = epsilon_end
        self._epsilon_decay = epsilon_decay
        self.reward_scale = reward_scale

        # Exploration related.
        self.explore_step = 0
        self.state = None  # Reset during training

        self.mapped_circ = CompositeGate()
        self._last_exec_cnt = 0
        self._last_action = None

        # Random data generator
        self.factory = DataFactory(
            topo=topo,
            max_gate_num=max_gate_num,
        )

        self.action_id_by_swap: Dict[Tuple[int, int], int] = {}
        self.swap_by_action_id: Dict[int, Tuple[int, int]] = {}
        self.action_num = 0
        self.register_topo(topo)
        assert self.topo is not None
        self._qubit_num = self.topo.qubit_number

    def register_topo(self, topo: Union[Layout, str]):
        if isinstance(topo, str):
            self.topo = self.factory.topo_map[topo]
        elif isinstance(topo, Layout):
            self.topo = topo
        else:
            raise TypeError("Only supports a layout name or Layout object.")

        swaps = [(edge.u, edge.v) for edge in self.topo]
        swaps.sort()
        for idx, swap in enumerate(swaps):
            self.action_id_by_swap[swap] = idx
            self.swap_by_action_id[idx] = swap
        self.action_num = len(self.action_id_by_swap)

    def reset_explore_state(self):
        self.state = self.factory.get_one()

    def _select_action(
        self,
        policy_net: NnMapping,
        policy_net_device: str,
    ):
        a = self.action_num

        # Chose an action based on policy_net
        data = PygBatch.from_data_list([self.state.circ_pyg_data]).to(
            policy_net_device
        )
        # data = self.state.circ_layered_matrices.to(policy_net_device)
        q_vec = policy_net(data).detach().cpu()
        q_vec = q_vec.view(a)  # [a]

        # Query action swap using action id
        idx = int(torch.argmax(q_vec))
        action = self.swap_by_action_id[idx]
        return action

    def select_action(
        self,
        policy_net: NnMapping,
        policy_net_device: str,
        epsilon_random: bool = False,
    ) -> Tuple[int, int]:
        """Select action based on given policy net.

        Args:
            policy_net (GnnMapping): A policy network which gives an action as output.
            policy_net_device (str): Policy net model device.
            epsilon_random (bool, optional): Whether to use epsilon-greedy strategy.
                If set True, there will be epsilon probability that this methods randomly select a viable action.
                Defaults to False.

        Returns:
            Tuple[int, int]: Selected action.
        """

        eps_threshold = self._epsilon_end + (
            self._epsilon_start - self._epsilon_end
        ) * math.exp(-1.0 * self.explore_step / self._epsilon_decay)
        if epsilon_random and random() <= eps_threshold:
            return self.state.biased_random_swap()
        else:
            return self._select_action(
                policy_net=policy_net,
                policy_net_device=policy_net_device,
            )

    def count_gate_num(self) -> int:
        return self.state.circ_info.count_gate()

    def take_action(
        self, action: Tuple[int, int]
    ) -> Tuple[State, Union[State, None], float, bool]:
        """Take given action on trainer's state.

        Args:
            action (Tuple[int, int]): Swap gate used on current topology.

        Returns:
            Tuple[State, Union[State, None], float, bool]: Tuple of (Previous State, Next State, Reward, Terminated).
        """
        self.explore_step += 1
        u, v = action
        graph = self.state.topo_info.topo_graph
        scale = self.reward_scale
        if not graph.has_edge(u, v):
            reward = -scale
            prev_state = self.state
            # next_state = None
            next_state = self.state
            self.state = next_state
            return prev_state, next_state, reward, True

        with self.mapped_circ:
            Swap & [u, v]

        next_logic2phy = copy.deepcopy(self.state.logic2phy)
        next_phy2logic = copy.deepcopy(self.state.phy2logic)
        # Current (u, v) are physical qubit labels
        next_phy2logic[u], next_phy2logic[v] = next_phy2logic[v], next_phy2logic[u]
        lu, lv = next_phy2logic[u], next_phy2logic[v]
        next_logic2phy[lu], next_logic2phy[lv] = next_logic2phy[lv], next_logic2phy[lu]
        next_circ_state = self.state.circ_info.copy()
        action_penalty = 0
        reward = action_penalty
        # Execute as many as possible
        cnt = next_circ_state.eager_exec(
            logic2phy=next_logic2phy,
            topo_graph=self.state.topo_info.topo_graph,
            physical_circ=self.mapped_circ,
        )
        self._last_action = action
        self._last_exec_cnt = cnt
        reward += cnt * scale

        terminated = next_circ_state.count_gate() == 0
        if terminated:
            prev_state = self.state
            # next_state = None
            next_state = self.state
            self.state = next_state
            return prev_state, next_state, reward, True

        next_state = State(
            circ_info=next_circ_state,
            topo=self.state.topo_info.topo,
            logic2phy=next_logic2phy,
            phy2logic=next_phy2logic,
        )
        prev_state = self.state
        self.state = next_state
        return prev_state, next_state, reward, False

    def map_all(
        self,
        max_gate_num: int,
        circ: CircuitBased,
        layout: Layout,
        policy_net: NnMapping,
        policy_net_device: str,
        cutoff: int = None,
    ) -> Tuple[CompositeGate, CompositeGate]:
        """Map given circuit to topology layout. Note that this methods will change internal exploration state.

        Args:
            max_qubit_num (int): Maximal qubit number after padding.
            max_gate_num (int): Maximal gate number after padding.
            circ (CircuitBased): Circuit to be mapped.
            layout (Layout): Topology layout.
            policy_net (GnnMapping): Policy network.
            policy_net_device (str): Policy network device.
            cutoff (int, optional): The maximal steps of execution. If the policy network cannot
                stop with in `cutoff` steps, force it to early stop. Defaults to None.

        Returns:
            Tuple[CompositeGate, CompositeGate]: Mapped circuit, remained circuit.
        """
        logic2phy = [i for i in range(layout.qubit_number)]
        agent = Agent(topo=layout, max_gate_num=max_gate_num)
        topo_graph = agent.factory.get_topo_graph(topo=layout)
        circ_state = CircuitInfo(circ=circ, max_gate_num=policy_net._max_gate_num)

        circ_state.eager_exec(
            logic2phy=logic2phy, topo_graph=topo_graph, physical_circ=agent.mapped_circ
        )
        agent.state = State(
            circ_info=circ_state,
            topo=layout,
            logic2phy=logic2phy,
        )

        terminated = circ_state.count_gate() == 0
        step = 0
        with torch.no_grad():
            while not terminated:
                action = agent.select_action(
                    policy_net=policy_net,
                    policy_net_device=policy_net_device,
                )
                _, _, _, terminated = agent.take_action(action=action)
                step += 1
                if cutoff is not None and step >= cutoff:
                    break
            return agent.mapped_circ, agent.state.remained_circ()
