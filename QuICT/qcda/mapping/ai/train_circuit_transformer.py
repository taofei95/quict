import math
import os
import os.path as osp
from collections import deque
from random import choice, randint, random, sample
from time import time
from typing import List, Set, Tuple, Union
from copy import deepcopy, copy

import torch
import torch.nn as nn
import torch.optim as optim
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

    def _reset_explore_state(self):
        print("Resetting exploration status...")
        start_time = time()
        self._state = State(*self._data_factory.get_one())
        # Move to GPU if needed
        self._state.x = self._state.x.to(self._device)
        self._state.spacial_encoding = self._state.spacial_encoding.to(self._device)
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
        graph = self._data_factory.topo_graph_map[self._state.topo_name]
        edges = self._data_factory.topo_edge_map[self._state.topo_name]
        if sample > eps_threshold:
            # Chose an action based on policy_net
            q_mat = self._policy_net(self._state.x, self._state.spacial_encoding)
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
        next_mapping = deepcopy(self._state.cur_mapping)
        next_mapping[u], next_mapping[v] = next_mapping[v], next_mapping[u]
        next_layered_circ = self._data_factory.remap_layered_circ(
            self._state.layered_circ, next_mapping
        )
        topo_dist = self._data_factory.topo_dist_map[self._state.topo_name]
        se_cpy = self._state.spacial_encoding.detach().clone()
        remove_layer_num = 0
        prev_layer_num = len(self._state.layered_circ)
        max_q = self._max_qubit_num
        # max_l = self._max_layer_num
        reward = 0.0
        for layer_idx, layer in enumerate(next_layered_circ):
            offset = (layer_idx + 1) * max_q
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

        # TODO: Avoid construct a brand new graph distance matrix
        next_spacial_encoding = torch.zeros_like(
            se_cpy, dtype=torch.int, device=self._device
        )
        remain_layer_start = remove_layer_num
        remain_layer_end = prev_layer_num
        remain_layer_num = remain_layer_end - remain_layer_start
        assert remain_layer_num == len(next_layered_circ)

        next_spacial_encoding[:max_q, :max_q] = se_cpy[:max_q, :max_q]
        next_spacial_encoding[:max_q, max_q : (remain_layer_num + 1) * max_q] = se_cpy[
            :max_q, (remain_layer_start + 1) * max_q : (remain_layer_end + 1) * max_q
        ]
        next_spacial_encoding[max_q : (remain_layer_num + 1) * max_q, :max_q] = se_cpy[
            (remain_layer_start + 1) * max_q : (remain_layer_end + 1) * max_q, :max_q
        ]
        next_spacial_encoding[
            max_q : (remain_layer_num + 1) * max_q,
            max_q : (remain_layer_num + 1) * max_q,
        ] = se_cpy[
            (remain_layer_start + 1) * max_q : (remain_layer_end + 1) * max_q,
            (remain_layer_start + 1) * max_q : (remain_layer_end + 1) * max_q,
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

        xs = torch.stack([state.x for state in states])
        spacial_encodings = torch.stack([state.spacial_encoding for state in states])
        rewards = torch.tensor(rewards, device=self._device)

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, next_states)),
            device=self._device,
            dtype=torch.bool,
        )
        non_final_next_xs = torch.stack(
            [state.x for state in next_states if state is not None]
        )
        non_final_next_spacial_encodings = torch.stack(
            [state.spacial_encoding for state in next_states if state is not None]
        )

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

    def train_one_epoch(self):
        self._reset_explore_state()
        observe_period = 10
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


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
