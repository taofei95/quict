import torch
import numpy as np

from QuICT.algorithm.quantum_machine_learning.utils import Ansatz
from QuICT.algorithm.quantum_machine_learning.utils.gate_tensor import *
from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.tools.exception.algorithm import *


class MCRy:
    def __init__(self, device=torch.device("cuda:0")):
        self._device = device

    def __call__(self, control=list, target=int, param=torch.Tensor()):
        n_control = len(control)
        assert n_control >= 2
        theta = param / (1 << n_control - 1)
