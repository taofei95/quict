import io
import math
import os
import os.path as osp
from collections import deque
from copy import copy, deepcopy
from random import choice, randint, random, sample
from time import time
from typing import List, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as TvT
from matplotlib import pyplot as plt
from QuICT.core import *
from QuICT.core.utils import GateType
from QuICT.qcda.mapping.ai.gtdqn import GraphTransformerDeepQNetwork
from QuICT.qcda.mapping.ai.transformer_data_factory import CircuitTransformerDataFactory
from torch.utils.tensorboard import SummaryWriter


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

    def __iter__(self):
        return iter((self.state, self.action, self.next_state, self.reward))


class ReplayMemory:
    def __init__(self, capacity: int) -> None:
        self._memory = deque([], maxlen=capacity)

    def push(self, transition: Transition):
        self._memory.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return sample(self._memory, batch_size)

    def __len__(self) -> int:
        return len(self._memory)


class Trainer:
    def __init__(
        self,
        max_qubit_num: int = 30,
        max_layer_num: int = 60,
        inner_feat_dim: int = 30,
        head: int = 3,
        num_attn_layer: int = 6,
        gamma: float = 0.99,
        replay_pool_size: int = 10000,
        batch_size: int = 16,
        total_epoch: int = 200,
        explore_period: int = 800,
        target_update_period: int = 200,
        epsilon_start: float = 0.9,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 100.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        model_path: str = None,
        log_dir: str = None,
    ) -> None:
        print("Initializing trainer...")

        # Copy values in.
        self._max_qubit_num = max_qubit_num
        self._max_layer_num = max_layer_num
        self._gamma = gamma
        self._batch_size = batch_size
        self._total_epoch = total_epoch
        self._explore_period = explore_period
        self._target_update_period = target_update_period
        self._epsilon_start = epsilon_start
        self._epsilon_end = epsilon_end
        self._epsilon_decay = epsilon_decay
        self._device = device

        # Experience replay memory pool
        print("Preparing experience pool...")
        self._replay = ReplayMemory(replay_pool_size)

        # Random data generator
        print("Building data factory...")
        self._data_factory = CircuitTransformerDataFactory(
            max_qubit_num=max_qubit_num,
            max_layer_num=max_layer_num,
            device=device,
        )
        self._data_factory._reset_topo_attr_cache()

        # Exploration related.
        self._explore_step = 0
        self._state = None  # Reset during training

        # DQN
        print("Resetting policy & target model...")
        self._policy_net = GraphTransformerDeepQNetwork(
            max_qubit_num=max_qubit_num,
            max_layer_num=max_layer_num,
            inner_feat_dim=inner_feat_dim,
            head=head,
            num_attn_layer=num_attn_layer,
        ).to(device=device)
        self._target_net = GraphTransformerDeepQNetwork(
            max_qubit_num=max_qubit_num,
            max_layer_num=max_layer_num,
            inner_feat_dim=inner_feat_dim,
            head=head,
            num_attn_layer=num_attn_layer,
        ).to(device=device)
        # Guarantee they two have the same parameter values.
        self._target_net.load_state_dict(self._policy_net.state_dict())
        self._loss_fn = nn.SmoothL1Loss()
        self._optimizer = optim.RMSprop(self._policy_net.parameters())

        # Prepare path to save model files during training
        print("Preparing data directory...")
        if model_path is None:
            model_path = osp.dirname(osp.abspath(__file__))
            model_path = osp.join(model_path, "model_graphormer")
        if not osp.exists(model_path):
            os.makedirs(model_path)
        self._model_path = model_path

        # Prepare summary writer and its logging directory
        if log_dir is None:
            log_dir = osp.dirname(osp.abspath(__file__))
            log_dir = osp.join(log_dir, "torch_runs")
            log_dir = osp.join(log_dir, "circuit_graphormer")
        if not osp.exists(log_dir):
            os.makedirs(log_dir)
        self._writer = SummaryWriter(log_dir=log_dir)

        # Validation circuits examples
        v_size = 3
        v_topo_names = sample(self._data_factory.topo_names, k=v_size)
        self._v_circs = []
        for topo_name in v_topo_names:
            q = self._data_factory.topo_qubit_num_map[topo_name]
            circ = Circuit(q)
            circ.random_append(rand_size=q * 5, typelist=[GateType.cx])
            x = self._data_factory.get_x(q)
            layered_circ, success = self._data_factory.get_layered_circ(circ)
            assert success
            circ_edges = self._data_factory.get_circ_edges(
                layered_circ, self._data_factory.topo_dist_map[topo_name]
            )
            spacial_encoding = self._data_factory.get_spacial_encoding(circ_edges)
            self._v_circs.append((topo_name, x, spacial_encoding))

    def _reset_explore_state(self):
        print("Resetting exploration status...")
        start_time = time()
        self._state = State(*self._data_factory.get_one())
        self._state.x = self._state.x.to("cpu")
        self._state.spacial_encoding = self._state.spacial_encoding.to("cpu")
        end_time = time()
        duration = end_time - start_time
        ps = len(self._replay)
        print(
            f"Reset fished within {duration:0.4f}s. Current experience pool size: {ps}"
        )

    def _select_action(self) -> Tuple[int, int]:
        eps_threshold = self._epsilon_end + (
            self._epsilon_start - self._epsilon_end
        ) * math.exp(-1.0 * self._explore_step / self._epsilon_decay)
        self._explore_step += 1
        sample = random()
        edges = self._data_factory.topo_edge_map[self._state.topo_name]
        if sample > eps_threshold:
            # Chose an action based on policy_net
            x = self._state.x.to(self._device)
            se = self._state.spacial_encoding.to(self._device)
            q_mat = self._policy_net(x, se).detach().cpu()
            q_mat = q_mat.view(self._max_qubit_num, self._max_qubit_num)
            topo_qubit_num = self._data_factory.topo_qubit_num_map[
                self._state.topo_name
            ]
            # Use a mask matrix to filter out unwanted qubit pairs.
            q_mat = (
                q_mat[:topo_qubit_num, :topo_qubit_num]
                * self._data_factory.topo_mask_map[self._state.topo_name]
            )
            pos = int(torch.argmax(q_mat))
            del q_mat
            u, v = pos // topo_qubit_num, pos % topo_qubit_num
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
            reward = -10.0
            next_state = self._state
            return next_state, reward, False

        next_mapping = deepcopy(self._state.cur_mapping)
        next_mapping[u], next_mapping[v] = next_mapping[v], next_mapping[u]
        next_layered_circ = self._data_factory.remap_layered_circ(
            self._state.layered_circ, next_mapping
        )
        topo_dist = self._data_factory.topo_dist_map[self._state.topo_name]
        se_cpy = self._state.spacial_encoding.detach().clone()
        remove_layer_num = 0
        prev_layer_num = len(self._state.layered_circ)
        _q = self._max_qubit_num
        # max_l = self._max_layer_num
        reward = 0.0
        for layer_idx, layer in enumerate(next_layered_circ):
            offset = (layer_idx + 1) * _q
            for x, y in copy(layer):
                if topo_dist[x][y] == 1:
                    layer.remove((x, y))
                    se_cpy[x + offset][y + offset] = 0
                    se_cpy[y + offset][x + offset] = 0
                    reward += 10.0
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

        next_spacial_encoding = torch.zeros_like(se_cpy, dtype=torch.int, device="cpu")
        _s = remove_layer_num  # Remain layer id start
        _e = prev_layer_num  # Remain layer id end
        _n = _e - _s  # Remain layer number
        assert _n == len(next_layered_circ)
        
        # Vertices in the first layer cannot reach each other. Keep the distance as 0.
        # next_spacial_encoding[:_q, :_q] = se_cpy[:_q, :_q]

        # Add none zero entries first remained layer forwards with 1. Use this part as the first virtual layer distances forwards.
        _tmp = se_cpy[(_s + 1) * _q : (_s + 2) * _q, (_s + 1) * _q : (_e + 1) * _q]
        _mask = _tmp.clamp(min=0, max=1)
        next_spacial_encoding[:_q, _q : (_n + 1) * _q] = (_tmp + 1) * _mask

        # No vertices can go back to the first layer. So keeping the distance as 0 is enough.
        # next_spacial_encoding[_q : (_n + 1) * _q, :_q] = 0

        # Copy bottom-right corner.
        next_spacial_encoding[_q : (_n + 1) * _q, _q : (_n + 1) * _q,] = se_cpy[
            (_s + 1) * _q : (_e + 1) * _q,
            (_s + 1) * _q : (_e + 1) * _q,
        ]

        next_state = State(
            layered_circ=next_layered_circ,
            topo_name=self._state.topo_name,
            x=next_x,
            spacial_encoding=next_spacial_encoding,
            cur_mapping=next_mapping,
        )

        return next_state, reward, False

    def _optimize_model(self) -> Union[None, float]:
        if len(self._replay) < self._batch_size:
            print(
                f"Experience pool is too small({len(self._replay)}). Keep exploring..."
            )
            return None
        transitions = self._replay.sample(self._batch_size)
        states, actions, next_states, rewards = zip(*transitions)

        actions = torch.tensor(
            [[u * self._max_qubit_num + v] for u, v in actions],
            dtype=torch.int64,
            device=self._device,
        )  # [B, 1]

        xs = torch.stack([state.x for state in states]).to(self._device)
        spacial_encodings = torch.stack(
            [state.spacial_encoding for state in states]
        ).to(self._device)
        rewards = torch.tensor(rewards, device=self._device)

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, next_states)),
            device=self._device,
            dtype=torch.bool,
        )
        non_final_next_xs = torch.stack(
            [state.x for state in next_states if state is not None]
        ).to(self._device)
        non_final_next_spacial_encodings = torch.stack(
            [state.spacial_encoding for state in next_states if state is not None]
        ).to(self._device)

        state_action_values = (
            self._policy_net(xs, spacial_encodings).gather(1, actions).squeeze()
        )
        next_state_values = torch.zeros(self._batch_size, device=self._device)
        next_state_values[non_final_mask] = (
            self._target_net(non_final_next_xs, non_final_next_spacial_encodings)
            .max(1)[0]
            .detach()
        )
        expected_state_action_values = (next_state_values * self._gamma) + rewards

        loss = self._loss_fn(state_action_values, expected_state_action_values)

        self._optimizer.zero_grad()
        loss.backward()
        loss_val = loss.item()
        for param in self._policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self._optimizer.step()
        return loss_val

    def _draw_v_heat_maps(self, timestamp: int):
        for topo_name, x, spacial_encoding in self._v_circs:
            q_mat = self._policy_net(x, spacial_encoding)
            q_mat = q_mat.detach().cpu().numpy()
            q_mat = np.reshape(q_mat, (self._max_qubit_num, self._max_qubit_num))
            topo_qubit_num = self._data_factory.topo_qubit_num_map[topo_name]
            mask = self._data_factory.topo_mask_map[topo_name].numpy()
            q_mat = q_mat[:topo_qubit_num, :topo_qubit_num] * mask
            # Normalize q_mat
            q_mat = np.clip(q_mat, 0, None)
            q_mat = (q_mat - np.min(q_mat)) / np.ptp(q_mat)
            e_mat = self._data_factory.topo_edge_mat_map[topo_name]
            fig, (ax1, ax2) = plt.subplots(ncols=2)
            im1 = ax1.matshow(q_mat, interpolation=None)
            im2 = ax2.matshow(e_mat, interpolation=None)
            fig.colorbar(im1, ax=ax1)
            fig.colorbar(im2, ax=ax2)
            self._writer.add_figure(f"{topo_name} policy attention", fig, timestamp)
            del q_mat

    def train_one_epoch(self):
        self._reset_explore_state()
        observe_period = 20
        running_loss = 0.0
        running_reward = 0.0
        last_stamp = time()
        for i in range(self._explore_period):
            if self._state is None:
                # Search finishes early.
                print("State terminates. Search stops early.")
                break

            action = self._select_action()
            next_state, reward, terminated = self._take_action(action=action)
            running_reward += reward

            # Put this transition into experience replay pool.
            self._replay.push(
                Transition(
                    state=self._state,
                    action=action,
                    next_state=next_state,
                    reward=reward,
                )
            )
            # Update state.
            self._state = next_state

            cur_loss = self._optimize_model()
            if cur_loss is None:
                continue
            running_loss += cur_loss

            # Update target net every C steps.
            if (self._explore_step + 1) % self._target_update_period == 0:
                print(
                    f"    Already explored {self._explore_step + 1} steps. Updating model..."
                )
                self._target_net.load_state_dict(self._policy_net.state_dict())

            self._writer.add_scalar("Loss", cur_loss, self._explore_step)
            self._writer.add_scalar("Reward", reward, self._explore_step)
            if (i + 1) % observe_period == 0:
                cur = time()
                duration = cur - last_stamp
                last_stamp = cur
                rate = duration / observe_period
                running_loss /= observe_period
                running_reward /= observe_period
                print(
                    f"    [{i+1}] loss: {running_loss:0.4f}, average reward: {running_reward:0.4f}, explore rate: {rate:0.4f} s/action"
                )
                running_reward = 0.0
                running_loss = 0.0

    def train(self):
        print(f"Training on {self._device}...\n")
        for epoch_id in range(1, 1 + self._total_epoch):
            print(f"Epoch {epoch_id}:")
            self.train_one_epoch()
            print()
            self._draw_v_heat_maps(epoch_id + 1)


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
