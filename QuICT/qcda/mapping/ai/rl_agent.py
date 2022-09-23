import copy
import math
from random import choice, random
from typing import Dict, List, Tuple, Union

import networkx as nx
import numpy as np
import torch
from QuICT.core import *
from QuICT.core.gate import *
from QuICT.core.utils import CircuitBased
from QuICT.qcda.mapping.ai.data_factory import (
    CircuitState,
    DataFactory,
    State,
    Transition,
)
from QuICT.qcda.mapping.ai.gnn_mapping import GnnMapping
from torch_geometric.data import Batch as PygBatch


class Agent:
    def __init__(
        self,
        max_qubit_num: int,
        max_gate_num: int,
        epsilon_start: float = 0.9,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 100.0,
        reward_scale: float = 10.0,
    ) -> None:
        # Copy values in.
        self._max_qubit_num = max_qubit_num
        self._max_gate_num = max_gate_num
        self._epsilon_start = epsilon_start
        self._epsilon_end = epsilon_end
        self._epsilon_decay = epsilon_decay
        self.reward_scale = reward_scale

        # Exploration related.
        self.explore_step = 0
        self.state = None  # Reset during training

        # Random data generator
        self.factory = DataFactory(
            max_qubit_num=max_qubit_num,
            max_gate_num=max_gate_num,
        )

        self.mapped_circ = CompositeGate()

    def reset_explore_state(self, topo_name: str = None):
        self.state = self.factory.get_one(topo_name=topo_name)

    def _select_action(
        self,
        policy_net: GnnMapping,
        policy_net_device: str,
    ):
        max_q = self._max_qubit_num
        q = self.state.topo.qubit_number

        # Chose an action based on policy_net
        circ_pyg = PygBatch.from_data_list([self.state.circ_pyg_data]).to(
            policy_net_device
        )
        topo_pyg = PygBatch.from_data_list([self.state.topo_pyg_data]).to(
            policy_net_device
        )
        attn = policy_net(circ_pyg, topo_pyg).detach().cpu()
        attn = attn.view(max_q, max_q)

        # Use a mask matrix to filter out unwanted qubit pairs.
        mask = self.state.topo_mask[:q, :q]
        attn: torch.Tensor = attn[:q, :q]
        # https://discuss.pytorch.org/t/masked-argmax-in-pytorch/105341/2
        large = torch.finfo(attn.dtype).max
        # pos = int((attn - large * (1 - mask) - large * (1 - mask)).argmax())
        pos = int((attn - large * (1 - mask)).argmax())
        u, v = pos // q, pos % q
        assert u < q and v < q and mask[u][v] > 0
        return u, v

    def select_action(
        self,
        policy_net: GnnMapping,
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
        return self.state.circ_state.count_gate()

    def take_action(
        self, action: Tuple[int, int], construct_gate: bool = False
    ) -> Tuple[State, Union[State, None], float, bool]:
        """Take given action on trainer's state.

        Args:
            action (Tuple[int, int]): Swap gate used on current topology.
            logical_circ (CircuitBased): If not None, will perform action and remove some gates on given logical circuits.
                Some gates may be inserted into the agent's itself's `mapped_circ`.

        Returns:
            Tuple[State, Union[State, None], float, bool]: Tuple of (Previous State, Next State, Reward, Terminated).
        """
        self.explore_step += 1
        u, v = action
        graph = self.state.topo_graph
        scale = self.reward_scale
        if not graph.has_edge(u, v):
            reward = -scale
            prev_state = self.state
            # next_state = None
            next_state = self.state
            self.state = next_state
            return prev_state, next_state, reward, True

        if construct_gate:
            with self.mapped_circ:
                Swap & [u, v]

        next_logic2phy = copy.deepcopy(self.state.logic2phy)
        next_logic2phy[u], next_logic2phy[v] = next_logic2phy[v], next_logic2phy[u]
        next_circ_state = self.state.circ_state.copy()
        action_penalty = -1
        reward = action_penalty

        physical_circ = self.mapped_circ if construct_gate else None
        cnt = next_circ_state.eager_exec(
            logic2phy=next_logic2phy,
            topo_graph=self.state.topo_graph,
            physical_circ=physical_circ,
        )
        reward += cnt * scale

        terminated = next_circ_state.count_gate() == 0
        if terminated:
            prev_state = self.state
            # next_state = None
            next_state = self.state
            self.state = next_state
            return prev_state, next_state, reward, True

        next_circ_pyg = next_circ_state.to_pyg(next_logic2phy)

        next_state = State(
            circ_graph=next_circ_state,
            topo=self.state.topo,
            topo_mask=self.state.topo_mask,
            topo_graph=self.state.topo_graph,
            topo_dist=self.state.topo_dist,
            topo_edges=self.state.topo_edges,
            circ_pyg_data=next_circ_pyg,
            topo_pyg_data=self.state.topo_pyg_data,
            logic2phy=next_logic2phy,
            # phy2logic=next_phy2logic,
        )
        prev_state = self.state
        self.state = next_state
        return prev_state, next_state, reward, False

    @classmethod
    def map_all(
        cls,
        max_qubit_num: int,
        max_gate_num: int,
        circ: CircuitBased,
        layout: Layout,
        policy_net: GnnMapping,
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
        logic2phy = [i for i in range(max_qubit_num)]
        agent = Agent(max_qubit_num=max_qubit_num, max_gate_num=max_gate_num)
        topo_graph = agent.factory._get_topo_graph(topo=layout)
        topo_dist = agent.factory._get_topo_dist(topo_graph=topo_graph)
        topo_mask = agent.factory._get_topo_mask(topo_graph=topo_graph)
        topo_edges = agent.factory._get_topo_edges(topo_graph=topo_graph)
        circ_state = CircuitState(circ=circ, max_gate_num=policy_net._max_gate_num)
        circ_pyg = circ_state.to_pyg(logic2phy)
        topo_pyg = agent.factory.get_topo_pyg(topo_graph=topo_graph)

        circ_state.eager_exec(
            logic2phy=logic2phy, topo_graph=topo_graph, physical_circ=agent.mapped_circ
        )
        agent.state = State(
            circ_graph=circ_state,
            topo=layout,
            topo_mask=topo_mask,
            topo_graph=topo_graph,
            topo_dist=topo_dist,
            topo_edges=topo_edges,
            circ_pyg_data=circ_pyg,
            topo_pyg_data=topo_pyg,
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
                _, _, _, terminated = agent.take_action(
                    action=action, construct_gate=True
                )
                step += 1
                if cutoff is not None and step >= cutoff:
                    break
            return agent.mapped_circ, agent.state.remained_circ()
