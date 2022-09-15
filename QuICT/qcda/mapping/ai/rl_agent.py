import copy
import math
from random import choice, random
from typing import List, Set, Tuple, Union

import networkx as nx
import numpy as np
import torch
from QuICT.core import *
from QuICT.core.gate import *
from QuICT.core.utils import CircuitBased
from QuICT.qcda.mapping.ai.data_factory import DataFactory
from QuICT.qcda.mapping.ai.gnn_mapping import GnnMapping


class State:
    def __init__(
        self,
        layered_circ: List[Set[Tuple[int, int]]],
        topo: Layout,
        topo_mask: torch.Tensor,
        topo_graph: nx.Graph,
        topo_dist: np.ndarray,
        topo_edges: Tuple[Tuple[int, int]],
        x: torch.LongTensor,
        edge_index: torch.LongTensor,
        logic2phy: List[int],
        phy2logic: List[int] = None,
    ) -> None:
        self.layered_circ = layered_circ
        self.topo = topo
        self.topo_mask = topo_mask
        self.topo_graph = topo_graph
        self.topo_dist = topo_dist
        self.topo_edges = topo_edges
        self.x = x
        self.edge_index = edge_index
        self.logic2phy = logic2phy
        """Logical to physical mapping
        """
        self.phy2logic = phy2logic
        """Physical to logical mapping
        """
        if phy2logic is None:
            self.phy2logic = [-1 for _ in range(len(logic2phy))]
            for i in range(len(logic2phy)):
                self.phy2logic[logic2phy[i]] = i


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
        self.explore_step = 0
        self.state = None  # Reset during training

        # Random data generator
        self.factory = DataFactory(
            max_qubit_num=max_qubit_num,
            max_layer_num=max_layer_num,
        )

        self.mapped_circ = CompositeGate()

    def reset_explore_state(self, topo_name: str = None):
        self.state = State(*self.factory.get_one(topo_name=topo_name))

    def _select_action(
        self,
        policy_net: GnnMapping,
        policy_net_device: str,
    ):
        max_q = self._max_qubit_num
        q = self.state.topo.qubit_number

        # Chose an action based on policy_net
        x = self.state.x.to(policy_net_device)
        edge_index = self.state.edge_index.to(policy_net_device)
        attn = policy_net(x, edge_index).detach().cpu()
        attn = attn.view(max_q, max_q)

        # Use a mask matrix to filter out unwanted qubit pairs.
        mask = self.state.topo_mask[:q, :q]
        attn: torch.Tensor = attn[:q, :q]
        # https://discuss.pytorch.org/t/masked-argmax-in-pytorch/105341/2
        large = torch.finfo(attn.dtype).max
        pos = int((attn - large * (1 - mask) - large * (1 - mask)).argmax())
        u, v = pos // q, pos % q
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
            edges = self.state.topo_edges
            return choice(edges)
        else:
            return self._select_action(
                policy_net=policy_net,
                policy_net_device=policy_net_device,
            )

    def first_layer_dist_sum(
        self, layered_circ: List[Set[Tuple[int, int]]], topo_dist: np.ndarray
    ) -> float:
        s = 0.0
        if len(layered_circ) == 0:
            return s
        for u, v in layered_circ[0]:
            s += topo_dist[u][v]
        s /= 2
        return s

    def count_gate_num(self) -> int:
        num = 0
        for layer in self.state.layered_circ:
            num += len(layer) // 2
        return num

    def take_action(
        self, action: Tuple[int, int], logical_circ: CircuitBased = None
    ) -> Tuple[State, Union[State, None], float, bool]:
        """Take given action on trainer's state.

        Args:
            action (Tuple[int, int]): Swap gate used on current topology.
            logical_circ (CircuitBased): If not None, will perform action and remove some gates on given logical circuits.
                Some gates may be inserted into the agent's itself's `mapped_circ`.

        Returns:
            Tuple[Union[State, None], bool]: Tuple of (Next State, Terminated).
        """
        self.explore_step += 1
        u, v = action
        graph = self.state.topo_graph
        scale = 1.0
        if not graph.has_edge(u, v):
            reward = 0.0
            next_state = None
            prev_state = self.state
            return prev_state, next_state, reward, True

        if logical_circ is not None:
            with self.mapped_circ:
                Swap & [u, v]

        next_logic2phy = copy.deepcopy(self.state.logic2phy)
        next_phy2logic = copy.deepcopy(self.state.phy2logic)
        _lu, _lv = next_phy2logic[v], next_phy2logic[u]
        next_phy2logic[u], next_phy2logic[v] = _lv, _lu
        # _lu, _lv = _lv, _lu
        next_logic2phy[_lu], next_logic2phy[_lv] = (
            next_logic2phy[_lv],
            next_logic2phy[_lu],
        )
        next_layered_circ = copy.deepcopy(self.state.layered_circ)
        topo_dist = self.state.topo_dist
        remove_layer_num = 0
        reward = 0.0
        for layer in next_layered_circ:
            for x, y in copy.copy(layer):
                _x = next_logic2phy[x]
                _y = next_logic2phy[y]
                if topo_dist[_x][_y] == 1:
                    layer.remove((x, y))
                    reward += scale / 2

                    if logical_circ is not None:
                        gate: BasicGate = logical_circ.gates.pop(0)
                        a, b = gate.cargs + gate.targs
                        with self.mapped_circ:
                            gate & [next_logic2phy[a], next_logic2phy[b]]

            if len(layer) > 0:
                break
            remove_layer_num += 1
        next_layered_circ = next_layered_circ[remove_layer_num:]

        # Check if there are only padded empty layers left.
        terminated = len(next_layered_circ) == 0
        if terminated:
            prev_state = self.state
            self.state = None
            return prev_state, None, reward, True

        # X for the same topology is always the same, so there's ne need to copy.
        next_x = self.state.x

        # Generate new edge_index
        topo_graph = self.state.topo_graph
        next_edge_index = self.factory.get_circ_edge_index(
            layered_circ=next_layered_circ,
            topo_graph=topo_graph,
            logic2phy=next_logic2phy,
        )

        next_state = State(
            layered_circ=next_layered_circ,
            topo=self.state.topo,
            topo_mask=self.state.topo_mask,
            topo_graph=self.state.topo_graph,
            topo_dist=self.state.topo_dist,
            topo_edges=self.state.topo_edges,
            x=next_x,
            edge_index=next_edge_index,
            logic2phy=next_logic2phy,
            phy2logic=next_phy2logic,
        )
        prev_state = self.state
        self.state = next_state
        return prev_state, next_state, reward, False

    def map_all(
        self,
        circ: CircuitBased,
        layout: Layout,
        policy_net: GnnMapping,
        policy_net_device: str,
        cutoff: int = None,
    ) -> CompositeGate:
        """Map given circuit to topology layout. Note that this methods will change internal exploration state.

        Args:
            circ (CircuitBased): Circuit to be mapped.
            layout (Layout): Topology layout.
            policy_net (GnnMapping): Policy network.
            policy_net_device (str): Policy network device.
            cutoff (int, optional): The maximal steps of execution. If the policy network cannot
                stop with in `cutoff` steps, force it to early stop. Defaults to None.

        Returns:
            CompositeGate: Mapped circuit.
        """
        cur_mapping = [i for i in range(self._max_qubit_num)]
        layered_circ, _ = self.factory.get_layered_circ(circ=circ)
        topo_graph = self.factory.get_topo_graph(topo=layout)
        topo_dist = self.factory.get_topo_dist(topo_graph=topo_graph)
        topo_mask = self.factory.get_topo_mask(topo_graph=topo_graph)
        topo_edges = self.factory.get_topo_edges(topo_graph=topo_graph)
        x = self.factory.get_x().to(device=policy_net_device)
        edge_index = self.factory.get_circ_edge_index(
            layered_circ=layered_circ,
            topo_graph=topo_graph,
            logic2phy=cur_mapping,
        ).to(device=policy_net_device)

        self.state = State(
            layered_circ=layered_circ,
            topo=layout,
            topo_mask=topo_mask,
            topo_graph=topo_graph,
            topo_dist=topo_dist,
            topo_edges=topo_edges,
            x=x,
            edge_index=edge_index,
            logic2phy=cur_mapping,
        )

        terminated = False
        step = 0
        with torch.no_grad():
            while not terminated:
                action = self.select_action(
                    policy_net=policy_net,
                    policy_net_device=policy_net_device,
                )
                _, _, _, terminated = self.take_action(action=action, logical_circ=circ)
                step += 1
                if cutoff is not None and step >= cutoff:
                    break
            return self.mapped_circ
