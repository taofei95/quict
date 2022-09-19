import numpy as np
import torch
from utils.hamiltonian import Hamiltonian
from torch.optim import Optimizer


class VQA:
    def __init__(
        self, hamiltonian: Hamiltonian, init_params: np.ndarray,
    ):
        self._hamiltonian = hamiltonian
        self._init_params = init_params

    def cal_expect(self):
        raise NotImplementedError

    def construct_ansatz(self):
        raise NotImplementedError

    def loss(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError
