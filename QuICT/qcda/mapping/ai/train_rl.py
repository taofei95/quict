#!/usr/bin/env python3

import os
import os.path as osp
import re
from collections import deque
from random import sample
from time import time
from typing import Dict, List, Union

import torch
import torch.nn as nn
import torch.optim as optim
from QuICT.core import *
from QuICT.core.gate.composite_gate import CompositeGate
from QuICT.qcda.mapping.ai.data_def import State, Transition
from QuICT.qcda.mapping.ai.nn_mapping import NnMapping
from QuICT.qcda.mapping.ai.rl_agent import Agent
from QuICT.tools.interface.qasm_interface import OPENQASMInterface
from torch.utils.tensorboard import SummaryWriter


def _wrap2circ(cg_or_circ: Union[Circuit, CompositeGate] = None):
    if cg_or_circ is None:
        return None
    elif isinstance(cg_or_circ, Circuit):
        return cg_or_circ
    elif isinstance(cg_or_circ, CompositeGate):
        circ = Circuit(cg_or_circ.width())
        circ.extend(cg_or_circ.gates)
        return circ
    else:
        raise TypeError("Only supports Circuit/CompositeGate")


class ReplayMemory:
    def __init__(self, capacity: int) -> None:
        self._memory = deque([], maxlen=capacity)

    def push(self, transition: Transition):
        self._memory.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return sample(self._memory, batch_size)

    def __len__(self) -> int:
        return len(self._memory)


class ValidationData:
    def __init__(
        self,
        circ: Circuit,
        topo: Layout,
        rl_mapped_circ: Union[Circuit, CompositeGate] = None,
        remained_circ: Union[Circuit, CompositeGate] = None,
    ) -> None:
        self.circ = circ
        self.rl_mapped_circ = rl_mapped_circ
        self.topo = topo
        self.remained_circ = remained_circ


class Trainer:
    def __init__(
        self,
        topo: Union[str, Layout],
        max_gate_num: int = 200,
        feat_dim: int = 100,
        gamma: float = 0.9,
        replay_pool_size: int = 20000,
        batch_size: int = 64,
        total_epoch: int = 2000,
        explore_period: int = 10000,
        target_update_period: int = 20,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        model_path: str = None,
        log_dir: str = None,
    ) -> None:
        print("Initializing trainer...")

        # Copy values in.
        self._max_gate_num = max_gate_num
        self._gamma = gamma
        self._batch_size = batch_size
        self._total_epoch = total_epoch
        self._explore_period = explore_period
        self._target_update_period = target_update_period
        self._device = device

        # Initialize Agent
        print("Initializing agent...")
        self._agent = Agent(
            topo=topo,
            max_gate_num=max_gate_num,
        )
        self._topo = self._agent.topo
        self._agent.factory._reset_attr_cache()

        print("Preparing policy & target networks...")
        self._policy_net = NnMapping(
            qubit_num=self._topo.qubit_number,
            max_gate_num=max_gate_num,
            feat_dim=feat_dim,
            action_num=self._agent.action_num,
        ).to(device=device)
        self._policy_net.train(True)
        self._target_net = NnMapping(
            qubit_num=self._topo.qubit_number,
            max_gate_num=max_gate_num,
            feat_dim=feat_dim,
            action_num=self._agent.action_num,
        ).to(device=device)
        self._target_net.train(False)
        # Guarantee they have the same parameter values.
        self._target_net.load_state_dict(self._policy_net.state_dict())

        # Experience replay memory pool
        print("Preparing experience pool...")
        self._replay = ReplayMemory(replay_pool_size)

        # Loss function & optimizer
        self._smooth_l1 = nn.SmoothL1Loss()
        self._optimizer = optim.RMSprop(self._policy_net.parameters(), lr=0.0001)

        # Prepare path to save model files during training
        print("Preparing model saving directory...")
        if model_path is None:
            model_path = osp.dirname(osp.abspath(__file__))
            model_path = osp.join(model_path, "model_rl_mapping")
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
        print("Preparing validation data...")
        self._v_data: List[ValidationData] = []
        self._mcts_num: Dict[str, int] = {}
        self._load_v_data()

        # best so far
        self._v_best_score = 1e9

    def _load_v_data(self):
        v_data_dir = osp.join(self._agent.factory._data_dir, "v_data")
        pattern = re.compile(r"(\w+)_(\d+).qasm")
        for _, _, file_names in os.walk(v_data_dir):
            for file_name in file_names:
                if file_name.startswith("mapped"):
                    continue

                file_path = osp.join(v_data_dir, file_name)
                # mapped_file_path = osp.join(v_data_dir, f"mapped_{file_name}")
                groups = pattern.match(file_name)
                topo_name = groups[1]
                if topo_name != self._topo.name:
                    continue
                topo = self._agent.factory.topo_map[topo_name]
                circ = OPENQASMInterface.load_file(file_path).circuit
                # mcts_mapped_circ = OPENQASMInterface.load_file(mapped_file_path).circuit
                self._v_data.append(
                    ValidationData(
                        circ=circ,
                        topo=topo,
                    )
                )
        self._v_data.sort(key=lambda x: len(x.circ.gates))

    def validate_model(self) -> List[ValidationData]:
        """Map all validation circuits to corresponding topologies.

        Returns:
            List[ValidationData]: Multiple instances.
        """
        for v_datum in self._v_data:
            cutoff = len(v_datum.circ.gates) * v_datum.circ.width()
            self._agent.mapped_circ.clean()
            result_circ, remained_circ = self._agent.map_all(
                max_gate_num=self._max_gate_num,
                circ=v_datum.circ,
                layout=v_datum.topo,
                policy_net=self._target_net,
                policy_net_device=self._device,
                cutoff=cutoff,
            )
            v_datum.rl_mapped_circ = _wrap2circ(result_circ)
            v_datum.remained_circ = _wrap2circ(remained_circ)
        return self._v_data

    def _optimize_model(self) -> Union[None, float]:
        if len(self._replay) < self._batch_size:
            print(
                f"Experience pool is too small({len(self._replay):2}). Keep exploring..."
            )
            return None
        transitions = self._replay.sample(self._batch_size)
        states, actions, next_states, rewards = zip(*transitions)

        actions = torch.tensor(
            [[self._agent.action_id_by_swap[action]] for action in actions],
            dtype=torch.long,
            device=self._device,
        )  # [B, 1]

        rewards = torch.tensor(rewards, device=self._device)

        data_list = [state.to_nn_data() for state in states]
        data_batch = State.batch_from_list(data_list=data_list, device=self._device)

        # Current Q estimation
        q_vec = self._policy_net(data_batch)  # [b, a]
        state_action_values = q_vec.gather(1, actions).squeeze()

        # Q* by Bellman Equation
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, next_states)),
            device=self._device,
            dtype=torch.bool,
        )
        non_final_data_list = [
            state.to_nn_data() for state in next_states if state is not None
        ]
        non_final_data_batch = State.batch_from_list(
            data_list=non_final_data_list, device=self._device
        )
        #
        # Use double DQN extension
        #
        # next_actions = (
        #     self._policy_net(non_final_data_batch)
        #     .clone()
        #     .detach()
        #     .max(1)[1]
        #     .view(-1, 1)
        # )
        # next_state_values = torch.zeros(self._batch_size, device=self._device)
        # next_state_values[non_final_mask] = (
        #     self._target_net(
        #         non_final_data_batch,
        #     )  # [b, a]
        #     .gather(1, next_actions)
        #     .squeeze()
        # )

        #
        # Use normal DQN
        #
        next_state_values = torch.zeros(self._batch_size, device=self._device)
        next_state_values[non_final_mask] = (
            self._target_net(
                non_final_data_batch,
            )  # [b, a]
            .clone()
            .detach()
            .max(1)[0]
        )

        expected_state_action_values = (next_state_values * self._gamma) + rewards

        loss = self._smooth_l1(state_action_values, expected_state_action_values)

        self._optimizer.zero_grad()
        loss.backward()
        loss_val = loss.item()
        self._optimizer.step()
        return loss_val

    def train_one_epoch(self):
        self._agent.reset_explore_state()
        observe_period = 100
        running_loss = 0.0
        running_reward = 0.0
        last_stamp = time()

        for i in range(self._explore_period):
            self._policy_net.train()
            action = self._agent.select_action(
                policy_net=self._policy_net,
                policy_net_device=self._device,
                epsilon_random=True,
            )
            prev_state, next_state, reward, terminated = self._agent.take_action(
                action=action
            )
            if terminated:
                next_state = None
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
                if terminated:
                    print("    State terminates. Resetting exploration")
                    self._agent.reset_explore_state()
                continue
            running_loss += cur_loss

            # Update target net every C steps.
            if (self._agent.explore_step + 1) % self._target_update_period == 0:
                self._target_net.load_state_dict(self._policy_net.state_dict())

            self._writer.add_scalar("Loss", cur_loss, self._agent.explore_step)
            self._writer.add_scalar("Reward", reward, self._agent.explore_step)

            if terminated:
                print("    State terminates. Resetting exploration")
                self._agent.reset_explore_state()

            if (i + 1) % observe_period == 0:
                cur = time()
                duration = cur - last_stamp
                last_stamp = cur
                rate = observe_period / duration
                running_loss /= observe_period
                running_reward /= observe_period
                gate_num = self._agent.count_gate_num()
                layout_name = self._agent.state.topo_info.topo.name
                print(
                    f"    [{i+1:<4}] loss: {running_loss:6.4f}, reward: {running_reward:4.2f}, "
                    + f"#gate: {gate_num}, explore rate: {rate:4.2f} action/s, layout name: {layout_name}"
                )
                running_reward = 0.0
                running_loss = 0.0

    def show_validation_results(self, results: List[ValidationData], epoch_id: int):
        print("[Validation Summary]")
        rl_num = {}
        original_num = {}
        for v_datum in results:
            topo_name = v_datum.topo.name
            rl_num[topo_name] = 0
            original_num[topo_name] = 0
        for idx, v_datum in enumerate(results):
            topo_name = v_datum.topo.name
            _rl_num = len(v_datum.rl_mapped_circ.gates)
            _original_num = len(v_datum.circ.gates)
            rl_num[topo_name] += _rl_num
            original_num[topo_name] += _original_num
            print(
                f"    #Gate in circuit: {_original_num}, #Gate by RL: {_rl_num} ({topo_name})"
            )
        if rl_num[topo_name] < self._v_best_score:
            self._v_best_score = rl_num[topo_name]
            if epoch_id > 5:
                torch.save(
                    self._target_net.state_dict(),
                    osp.join(self._model_path, f"model_epoch_{epoch_id}.pt"),
                )
            # if self._topo.qubit_number <= 16:
            #     for idx, v_datum in enumerate(results):
            #         rl_circ_fig = v_datum.rl_mapped_circ.draw(method="matp_silent")
            #         original_circ_fig = v_datum.circ.draw(method="matp_silent")
            #         rl_circ_fig.dpi = 75
            #         original_circ_fig.dpi = 75
            #         self._writer.add_figure(
            #             f"{topo_name}-{idx} (RL)", rl_circ_fig, epoch_id
            #         )
            #         self._writer.add_figure(
            #             f"{topo_name}-{idx}", original_circ_fig, epoch_id
            #         )
        self._writer.add_scalars(
            f"Validation Performance ({topo_name})",
            {
                "#Gate in circuit": original_num[topo_name],
                "#Gate by RL": rl_num[topo_name],
                "#Gate by RL (best)": self._v_best_score,
            },
            epoch_id + 1,
        )
        print()

    def train(self):
        print(f"Training on {self._device}...\n")
        for epoch_id in range(1, 1 + self._total_epoch):
            print(f"Epoch {epoch_id}:")
            self.train_one_epoch()
            # Epoch ends. Validate current model.
            print("Validating current model on some circuits...")
            results = self.validate_model()
            self.show_validation_results(results, epoch_id)


if __name__ == "__main__":
    topo = "grid_5x5"
    device = "cuda:1"
    trainer = Trainer(topo=topo, device=device)
    trainer.train()
