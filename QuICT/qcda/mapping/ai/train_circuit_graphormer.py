import os
import os.path as osp
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from QuICT.qcda.mapping.ai.circuit_graphormer import CircuitGraphormer
from QuICT.qcda.mapping.ai.dataset import MappingGraphormerDataset


class State:
    def __init__(self) -> None:
        pass


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class Trainer:
    def __init__(
        self,
        batch_size: int,
        total_epoch: int = 200,
        device: str = "cpu",
        model_path: str = None,
        log_dir: str = None,
    ) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self._device = device

        dataset = MappingGraphormerDataset()
        max_qubit_number, max_layer_num = dataset.metadata

        # TODO: parameterize this model.
        self._model = CircuitGraphormer(
            max_qubit_num=max_qubit_number,
            max_layer_num=max_layer_num,
            max_topology_diameter=max_qubit_number,
            feat_dim=30,
            head=3,
        ).to(device=device)

        if model_path is None:
            model_path = osp.dirname(osp.abspath(__file__))
            model_path = osp.join(model_path, "model_graphormer")
        if not osp.exists(model_path):
            os.makedirs(model_path)
        self._model_path = model_path

        self._total_epoch = total_epoch

        self._loss = nn.SmoothL1Loss()

        if log_dir is None:
            log_dir = osp.dirname(osp.abspath(__file__))
            log_dir = osp.join(log_dir, "torch_runs")
            log_dir = osp.join(log_dir, "circuit_graphormer")

        self._writer = SummaryWriter(log_dir=log_dir)

        print(f"Start training on {device}...")

    def train_one_epoch(self, epoch: int):
        pass
