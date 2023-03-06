from os import path as osp
from typing import Union

import torch

from QuICT.core import *
from QuICT.qcda.mapping.ai.data_def import DataFactory


class Config:
    def __init__(
        self,
        topo: Union[str, Layout] = "grid_4x4",
        max_gate_num: int = 300,
        feat_dim: int = 96,
        gamma: float = 0.95,
        replay_pool_size: int = 10_000_000,
        lr: float = 0.00001,
        batch_size: int = 64,
        total_epoch: int = 2000,
        explore_period: int = 10000,
        target_update_period: int = 10,
        model_sync_period: int = 10,
        device: str = "cuda:2" if torch.cuda.is_available() else "cpu",
        model_path: str = None,
        log_dir: str = None,
        epsilon_start: float = 0.95,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 500_000.0,
        reward_scale: float = 10.0,
        inference: bool = False,
        inference_model_dir: str = "./model",
        tag: str = "sage_gnrom_lnb_aggr",
    ) -> None:
        self.factory = DataFactory(topo=topo, max_gate_num=max_gate_num)

        self.topo = self.factory._cur_topo

        swaps = [(edge.u, edge.v) for edge in self.topo]
        swaps.sort()
        self.action_id_by_swap = {}
        self.swap_by_action_id = {}
        for idx, swap in enumerate(swaps):
            self.action_id_by_swap[swap] = idx
            self.swap_by_action_id[idx] = swap
        self.action_num = len(self.action_id_by_swap)

        self.max_gate_num = max_gate_num
        self.feat_dim = feat_dim
        self.gamma = gamma
        self.replay_pool_size = replay_pool_size
        self.lr = lr
        self.batch_size = batch_size
        self.total_epoch = total_epoch
        self.explore_period = explore_period
        self.target_update_period = target_update_period
        self.model_sync_period = model_sync_period
        self.device = device
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.reward_scale = reward_scale
        self.inference = inference
        self.inference_model_dir = inference_model_dir

        if model_path is None:
            model_path = osp.dirname(osp.abspath(__file__))
            model_path = osp.join(model_path, f"{self.topo.name}-model_rl_mapping")
        self.model_path = model_path

        if log_dir is None:
            log_dir = osp.dirname(osp.abspath(__file__))
            log_dir = osp.join(log_dir, "torch_runs")
            log_dir = osp.join(log_dir, "rl_mapping")
            log_dir = osp.join(log_dir, self.topo.name)
        self.log_dir = log_dir

        self.tag = tag
