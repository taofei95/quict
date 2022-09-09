import os
import os.path as osp
from collections import deque
from random import sample
from time import time
from typing import List, Union

import torch
import torch.nn as nn
import torch.optim as optim
from QuICT.qcda.mapping.ai.gnn_mapping import GnnMapping
from QuICT.qcda.mapping.ai.rl import Agent, Transition
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
        gamma: float = 0.99,
        replay_pool_size: int = 10000,
        batch_size: int = 32,
        total_epoch: int = 200,
        explore_period: int = 2000,
        target_update_period: int = 400,
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
        self._agent._data_factory._reset_attr_cache()

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
        state_action_values = (
            self._policy_net(batch.x, batch.edge_index).gather(1, actions).squeeze()
        )

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

        loss = self._loss_fn(state_action_values, expected_state_action_values)

        self._optimizer.zero_grad()
        loss.backward()
        loss_val = loss.item()
        for param in self._policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self._optimizer.step()
        return loss_val

    # def _draw_v_heat_maps(self, timestamp: int):
    #     for topo_name, x, spacial_encoding in self._v_circs:
    #         q_mat = self._policy_net(x, spacial_encoding)
    #         q_mat = q_mat.detach().cpu().numpy()
    #         q_mat = np.reshape(q_mat, (self._max_qubit_num, self._max_qubit_num))
    #         topo_qubit_num = self._data_factory.topo_qubit_num_map[topo_name]
    #         mask = self._data_factory.topo_mask_map[topo_name].numpy()
    #         q_mat = q_mat[:topo_qubit_num, :topo_qubit_num] * mask
    #         # Normalize q_mat
    #         q_mat = np.clip(q_mat, 0, None)
    #         q_mat = (q_mat - np.min(q_mat)) / np.ptp(q_mat)
    #         e_mat = self._data_factory.topo_edge_mat_map[topo_name]
    #         fig, (ax1, ax2) = plt.subplots(ncols=2)
    #         im1 = ax1.matshow(q_mat, interpolation=None)
    #         im2 = ax2.matshow(e_mat, interpolation=None)
    #         fig.colorbar(im1, ax=ax1)
    #         fig.colorbar(im2, ax=ax2)
    #         self._writer.add_figure(f"{topo_name} policy attention", fig, timestamp)
    #         del q_mat

    def train_one_epoch(self):
        self._agent.reset_explore_state()
        observe_period = 50
        running_loss = 0.0
        running_reward = 0.0
        last_stamp = time()

        for i in range(self._explore_period):
            if self._agent._state is None:
                # Search finishes early.
                print("State terminates. Search stops early.")
                break

            self._policy_net.train()
            action = self._agent.select_action(
                policy_net=self._policy_net,
                policy_net_device=self._device,
            )
            next_state, reward, terminated = self._agent._take_action(action=action)
            running_reward += reward

            # Put this transition into experience replay pool.
            self._replay.push(
                Transition(
                    state=self._agent._state,
                    action=action,
                    next_state=next_state,
                    reward=reward,
                )
            )
            # Update state.
            self._state = next_state

            cur_loss = self._optimize_model()
            self._policy_net.train(False)
            if cur_loss is None:
                continue
            running_loss += cur_loss

            # Update target net every C steps.
            if (self._agent._explore_step + 1) % self._target_update_period == 0:
                print(
                    f"    Already explored {self._agent._explore_step + 1} steps. Updating model..."
                )
                self._target_net.load_state_dict(self._policy_net.state_dict())

            self._writer.add_scalar("Loss", cur_loss, self._agent._explore_step)
            self._writer.add_scalar("Reward", reward, self._agent._explore_step)
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
            # self._draw_v_heat_maps(epoch_id + 1)


if __name__ == "__main__":
    # trainer = Trainer(device="cpu")
    trainer = Trainer()
    trainer.train()
