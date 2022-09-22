import os
import os.path as osp
import re
from collections import deque
from random import sample
from time import time
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from QuICT.core import *
from QuICT.core.gate.composite_gate import CompositeGate
from QuICT.qcda.mapping.ai.gnn_mapping import GnnMapping
from QuICT.qcda.mapping.ai.rl_agent import Agent, Transition
from QuICT.tools.interface.qasm_interface import OPENQASMInterface
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch as PygBatch
from matplotlib.figure import Figure


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
        mcts_mapped_circ: Union[Circuit, CompositeGate],
        rl_mapped_circ: Union[Circuit, CompositeGate] = None,
    ) -> None:
        self.circ = circ
        self.mcts_mapped_circ = mcts_mapped_circ
        self.rl_mapped_circ = rl_mapped_circ
        self.topo = topo


# TODO: Add MCTS prior experience


class Trainer:
    def __init__(
        self,
        max_qubit_num: int = 5,
        max_gate_num: int = 50,
        feat_dim: int = 50,
        gamma: float = 0.9,
        replay_pool_size: int = 20000,
        batch_size: int = 32,
        total_epoch: int = 2000,
        explore_period: int = 500,
        target_update_period: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        model_path: str = None,
        log_dir: str = None,
    ) -> None:
        print("Initializing trainer...")

        # Copy values in.
        self._max_qubit_num = max_qubit_num
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
            max_qubit_num=max_qubit_num,
            max_gate_num=max_gate_num,
            inner_feat_dim=feat_dim,
        )
        self._agent.factory._reset_attr_cache()

        # Experience replay memory pool
        print("Preparing experience pool...")
        self._replay = ReplayMemory(replay_pool_size)

        # DQN
        print("Resetting policy & target model...")
        self._policy_net = GnnMapping(
            max_qubit_num=max_qubit_num,
            max_gate_num=max_gate_num,
            feat_dim=feat_dim,
        ).to(device=device)
        self._policy_net.train(True)
        self._target_net = GnnMapping(
            max_qubit_num=max_qubit_num,
            max_gate_num=max_gate_num,
            feat_dim=feat_dim,
        ).to(device=device)
        self._target_net.train(False)

        # Guarantee they two have the same parameter values.
        self._target_net.load_state_dict(self._policy_net.state_dict())

        # Loss function & optimizer
        self._smooth_l1 = nn.SmoothL1Loss()
        self._optimizer = optim.RMSprop(self._policy_net.parameters(), lr=0.008)

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
        self._fill_v_data()

    def _fill_v_data(self):
        v_data_dir = osp.join(self._agent.factory._data_dir, "v_data")
        pattern = re.compile(r"(\w+)_(\d+).qasm")
        for _, _, file_names in os.walk(v_data_dir):
            for file_name in file_names:
                if file_name.startswith("mapped"):
                    continue

                file_path = osp.join(v_data_dir, file_name)
                mapped_file_path = osp.join(v_data_dir, f"mapped_{file_name}")
                groups = pattern.match(file_name)
                topo_name = groups[1]
                # TODO: Rotate topo
                if "lima" not in topo_name:
                    continue
                topo = self._agent.factory.topo_map[topo_name]
                circ = OPENQASMInterface.load_file(file_path).circuit
                mcts_mapped_circ = OPENQASMInterface.load_file(mapped_file_path).circuit
                self._v_data.append(
                    ValidationData(
                        circ=circ,
                        topo=topo,
                        mcts_mapped_circ=mcts_mapped_circ,
                    )
                )

    def validate_model(self) -> List[ValidationData]:
        """Map all validation circuits to corresponding topologies.

        Returns:
            List[ValidationData]: Multiple instances.
        """
        for v_datum in self._v_data:
            # cutoff = len(v_datum.circ.gates) * v_datum.circ.width()
            cutoff = 30
            self._agent.mapped_circ.clean()
            result_circ = self._agent.map_all(
                circ=v_datum.circ,
                layout=v_datum.topo,
                policy_net=self._target_net,
                policy_net_device=self._device,
                cutoff=cutoff,
            )
            v_datum.rl_mapped_circ = _wrap2circ(result_circ)
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
            [[u * self._max_qubit_num + v] for u, v in actions],
            dtype=torch.int64,
            device=self._device,
        )  # [B, 1]

        circ_data_list = [state.circ_pyg_data for state in states]
        circ_pyg = PygBatch.from_data_list(circ_data_list).to(self._device)
        topo_data_list = [state.topo_pyg_data for state in states]
        topo_pyg = PygBatch.from_data_list(topo_data_list).to(self._device)
        rewards = torch.tensor(rewards, device=self._device)

        # Current Q estimation
        attn_mat = self._policy_net(circ_pyg, topo_pyg)  # [b, q * q]
        state_action_values = attn_mat.gather(1, actions).squeeze()

        # Q* by Bellman Equation
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, next_states)),
            device=self._device,
            dtype=torch.bool,
        )
        non_final_circ_data_list = [
            state.circ_pyg_data for state in next_states if state is not None
        ]
        non_final_circ_pyg = PygBatch.from_data_list(non_final_circ_data_list).to(
            self._device
        )
        non_final_topo_data_list = [
            state.topo_pyg_data for state in next_states if state is not None
        ]
        non_final_topo_pyg = PygBatch.from_data_list(non_final_topo_data_list).to(
            self._device
        )
        next_state_values = torch.zeros(self._batch_size, device=self._device)
        next_state_values[non_final_mask] = (
            self._target_net(
                non_final_circ_pyg,
                non_final_topo_pyg,
            )  # [b, q * q]
            .clone()
            .detach()
            .max(1)[0]
        )
        expected_state_action_values = (next_state_values * self._gamma) + rewards

        loss_1 = self._smooth_l1(state_action_values, expected_state_action_values)

        # Empirical loss on output attention
        # b = self._batch_size
        # q = self._max_qubit_num
        # mask = [state.topo_mask for state in states]
        # mask = (
        #     (torch.ones(b, q, q) - torch.stack(mask)).detach().to(device=self._device)
        # )
        # zeros = torch.zeros(b, q, q).detach().to(device=self._device)
        # attn_mat_illegal = F.relu(attn_mat.view(b, q, q) * (1 - mask))
        # loss_2 = self._smooth_l1(attn_mat_illegal, zeros) * 0.1

        # loss = loss_1 + loss_2
        loss = loss_1

        self._optimizer.zero_grad()
        loss.backward()
        loss_val = loss.item()
        self._optimizer.step()
        return loss_val

    def _random_fill_replay(self):
        """Fill replay pool with one circuit for each topology."""
        for topo_name in self._agent.factory.topo_names:
            self._agent.reset_explore_state(topo_name=topo_name)
            with torch.no_grad():
                action = self._agent.select_action(
                    policy_net=self._policy_net,
                    policy_net_device=self._device,
                    epsilon_random=True,
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
                epsilon_random=True,
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
                layout_name = self._agent.state.topo.name
                print(
                    f"    [{i+1:<4}] loss: {running_loss:6.4f}, reward: {running_reward:4.2f}, "
                    + f"#gate: {gate_num}, explore rate: {rate:4.2f} action/s, layout name: {layout_name}"
                )
                running_reward = 0.0
                running_loss = 0.0

    def show_validation_results(self, results: List[ValidationData], epoch_id: int):
        print("[Validation Summary]")
        mcts_num = {}
        rl_num = {}
        for v_datum in results:
            topo_name = v_datum.topo.name
            mcts_num[topo_name] = 0
            rl_num[topo_name] = 0
        for idx, v_datum in enumerate(results):
            topo_name = v_datum.topo.name
            mcts_num[topo_name] += len(v_datum.mcts_mapped_circ.gates)
            rl_num[topo_name] += len(v_datum.rl_mapped_circ.gates)
            mcts_circ_fig = v_datum.mcts_mapped_circ.draw(method="matp_silent")
            rl_circ_fig = v_datum.rl_mapped_circ.draw(method="matp_silent")
            mcts_circ_fig.dpi = 70
            rl_circ_fig.dpi = 70
            self._writer.add_figure(
                f"{topo_name}-{idx} (MCTS)", mcts_circ_fig, epoch_id
            )
            self._writer.add_figure(f"{topo_name}-{idx} (RL)", rl_circ_fig, epoch_id)
            print(
                f"    #Gate by RL: {rl_num[topo_name]}, #Gate by MCTS: {mcts_num[topo_name]} ({topo_name})"
            )
            self._writer.add_scalars(
                f"Validation Performance ({topo_name})",
                {
                    "#Gate by RL": rl_num[topo_name],
                    "#Gate by MCTS": mcts_num[topo_name],
                },
                epoch_id + 1,
            )
        print()

    def train(self):
        print(f"Training on {self._device}...\n")
        # self._random_fill_replay()
        for epoch_id in range(1, 1 + self._total_epoch):
            print(f"Epoch {epoch_id}:")
            self.train_one_epoch()
            # Epoch ends. Validate current model.
            print("Validating current model on some circuits...")
            results = self.validate_model()
            self.show_validation_results(results, epoch_id)
            if epoch_id % 200 == 0 and epoch_id >= 200:
                torch.save(
                    self._target_net.state_dict(),
                    osp.join(self._model_path, f"model_epoch_{epoch_id}.pt"),
                )


if __name__ == "__main__":
    # trainer = Trainer(device="cpu")
    trainer = Trainer()
    trainer.train()
