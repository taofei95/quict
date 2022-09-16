from signal import raise_signal
import numpy as np
from utils.hamiltonian import Hamiltonian
from torch.optim import Optimizer


class VQA:
    def __init__(
        self,
        ansatz,
        hamiltonian: Hamiltonian,
        optimizer: Optimizer,
        init_params: np.ndarray,
    ):
        self._ansatz = ansatz
        self._hamiltonian = hamiltonian
        self._optimizer = optimizer
        self._init_params = init_params

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

