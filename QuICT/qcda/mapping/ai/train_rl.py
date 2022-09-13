import os
import os.path as osp
from collections import deque
from random import sample
from time import time
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from QuICT.qcda.mapping.ai.gnn_mapping import GnnMapping
from QuICT.qcda.mapping.ai.rl_agent import Agent, Transition
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch as PygBatch
from torch_geometric.data import Data as PygData


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
        inner_feat_dim: int = 50,
        gamma: float = 0.9,
        replay_pool_size: int = 10000,
        batch_size: int = 32,
        total_epoch: int = 200,
        explore_period: int = 2000,
        target_update_period: int = 10,
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
        self._device = device

        # Initialize Agent
        print("Initializing agent...")
        self._agent = Agent(
            max_qubit_num=max_qubit_num,
            max_layer_num=max_layer_num,
            inner_feat_dim=inner_feat_dim,
        )
        self._agent.factory._reset_attr_cache()

        # Experience replay memory pool
        print("Preparing experience pool...")
        self._replay = ReplayMemory(replay_pool_size)

        # DQN
        print("Resetting policy & target model...")
        self._policy_net = GnnMapping(
            max_qubit_num=max_qubit_num,
            max_layer_num=max_layer_num,
            feat_dim=inner_feat_dim,
        ).to(device=device)
        self._target_net = GnnMapping(
            max_qubit_num=max_qubit_num,
            max_layer_num=max_layer_num,
            feat_dim=inner_feat_dim,
        ).to(device=device)

        # Guarantee they two have the same parameter values.
        self._target_net.load_state_dict(self._policy_net.state_dict())

        # Loss function & optimizer
        self._smooth_l1 = nn.SmoothL1Loss()
        self._optimizer = optim.RMSprop(self._policy_net.parameters(), lr=0.001)

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
            log_dir = osp.join(log_dir, "rl_mapping")
        if not osp.exists(log_dir):
            os.makedirs(log_dir)
        self._writer = SummaryWriter(log_dir=log_dir)

        # Validation circuits
        self._v_data = []
        for _ in range(3):
            _, topo_name, x, edge_index, _ = self._agent.factory.get_one()
            self._v_data.append((topo_name, x, edge_index))

    def _optimize_model(self) -> Union[None, float]:
        if len(self._replay) < self._batch_size:
            print(
                f"Experience pool is too small({len(self._replay):2}). Keep exploring..."
            )
            return None
        transitions = self._replay.sample(self._batch_size)
        states, actions, next_states, rewards = zip(*transitions)

        actions = torch.tensor(
            [[u * self._max_qubit_num + v] for u, v in actions],
            dtype=torch.int64,
            device=self._device,
        )  # [B, 1]

        data_list = [
            PygData(x=state.x, edge_index=state.edge_index) for state in states
        ]
        batch = PygBatch.from_data_list(data_list=data_list).to(device=self._device)
        rewards = torch.tensor(rewards, device=self._device)

        # Current Q estimation
        attn_mat = self._policy_net(batch.x, batch.edge_index)
        state_action_values = attn_mat.gather(1, actions).squeeze()

        # Q* by Bellman Equation
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, next_states)),
            device=self._device,
            dtype=torch.bool,
        )
        non_final_data_list = [
            PygData(x=state.x, edge_index=state.edge_index)
            for state in next_states
            if state is not None
        ]
        non_final_batch = PygBatch.from_data_list(data_list=non_final_data_list).to(
            device=self._device
        )
        next_state_values = torch.zeros(self._batch_size, device=self._device)
        next_state_values[non_final_mask] = (
            self._target_net(
                non_final_batch.x, non_final_batch.edge_index
            )  # [b, n * n]
            .clone()
            .detach()
            .max(1)[0]
        )
        expected_state_action_values = (next_state_values * self._gamma) + rewards

        loss_1 = self._smooth_l1(state_action_values, expected_state_action_values)

        # Empirical loss on output attention
        b = self._batch_size
        q = self._max_qubit_num
        mask = [self._agent.factory.topo_mask_map[state.topo_name] for state in states]
        mask = (
            (torch.ones(b, q, q) - torch.stack(mask)).detach().to(device=self._device)
        )
        zeros = torch.zeros(b, q, q).detach().to(device=self._device)

        loss_2 = self._smooth_l1(attn_mat.view(b, q, q) * mask, zeros) * 0.005

        loss = loss_1 + loss_2

        self._optimizer.zero_grad()
        loss.backward()
        loss_val = loss.item()
        self._optimizer.step()
        return loss_val

    def _draw_v_heat_maps(self, timestamp: int):
        for idx, (topo_name, x, edge_index) in enumerate(self._v_data):
            q_mat = self._policy_net(x.to(self._device), edge_index.to(self._device))
            q_mat = q_mat.detach().cpu().numpy()
            q_mat = np.reshape(q_mat, (self._max_qubit_num, self._max_qubit_num))
            mask = self._agent.factory.topo_mask_map[topo_name].numpy()
            q_mat = q_mat * mask
            # Normalize q_mat
            q_mat = np.clip(q_mat, 0, None)
            q_mat = (q_mat - np.min(q_mat)) / (np.ptp(q_mat) + 1e-7)
            fig, (ax1, ax2) = plt.subplots(ncols=2)
            im1 = ax1.matshow(mask, interpolation=None)
            im2 = ax2.matshow(q_mat, interpolation=None)
            ax1.set_title(f"Topology {topo_name}")
            ax2.set_title("Policy Attention")
            fig.colorbar(im1, ax=ax1)
            fig.colorbar(im2, ax=ax2)
            self._writer.add_figure(f"Policy Attention Example {idx}", fig, timestamp)

    def _random_fill_replay(self):
        """Fill replay pool with one circuit for each topology."""
        for topo_name in self._agent.factory.topo_names:
            self._agent.reset_explore_state(topo_name=topo_name)
            with torch.no_grad():
                action = self._agent.select_action(
                    policy_net=self._policy_net,
                    policy_net_device=self._device,
                )
            prev_state, next_state, reward, terminated = self._agent.take_action(
                action=action
            )
            self._replay.push(
                Transition(
                    state=prev_state,
                    action=action,
                    next_state=next_state,
                    reward=reward,
                )
            )

    def train_one_epoch(self):
        self._agent.reset_explore_state()
        observe_period = 50
        running_loss = 0.0
        running_reward = 0.0
        last_stamp = time()

        for i in range(self._explore_period):
            self._policy_net.train()
            action = self._agent.select_action(
                policy_net=self._policy_net,
                policy_net_device=self._device,
            )
            prev_state, next_state, reward, terminated = self._agent.take_action(
                action=action
            )
            running_reward += reward

            # Put this transition into experience replay pool.
            self._replay.push(
                Transition(
                    state=prev_state,
                    action=action,
                    next_state=next_state,
                    reward=reward,
                )
            )

            cur_loss = self._optimize_model()
            self._policy_net.train(False)
            if cur_loss is None:
                continue
            running_loss += cur_loss

            # Update target net every C steps.
            if (self._agent.explore_step + 1) % self._target_update_period == 0:
                # print(
                #     f"    Already explored {self._agent._explore_step + 1} steps. Updating model..."
                # )
                self._target_net.load_state_dict(self._policy_net.state_dict())

            self._writer.add_scalar("Loss", cur_loss, self._agent.explore_step)
            self._writer.add_scalar("Reward", reward, self._agent.explore_step)

            if terminated:
                # Search finishes early.
                print("State terminates. Search stops early.")
                break

            if (i + 1) % observe_period == 0:
                cur = time()
                duration = cur - last_stamp
                last_stamp = cur
                rate = duration / observe_period
                running_loss /= observe_period
                running_reward /= observe_period
                layer_num = len(self._agent._state.layered_circ)
                gate_num = self._agent.count_gate_num()
                print(
                    f"    [{i+1}] loss: {running_loss:0.4f}, average reward: {running_reward:0.4f}, explore rate: {rate:0.4f} s/action, #gate: {gate_num}, #layer: {layer_num}"
                )
                running_reward = 0.0
                running_loss = 0.0

    def train(self):
        print(f"Training on {self._device}...\n")
        self._random_fill_replay()
        for epoch_id in range(1, 1 + self._total_epoch):
            print(f"Epoch {epoch_id}:")
            self.train_one_epoch()
            print()
            self._draw_v_heat_maps(epoch_id + 1)


if __name__ == "__main__":
    # trainer = Trainer(device="cpu")
    trainer = Trainer()
    trainer.train()
