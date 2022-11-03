import torch
import torch.nn as nn
import torch.nn.functional as F

from QuICT.algorithm.quantum_machine_learning.ansatz_library import QNNLayer
from QuICT.algorithm.quantum_machine_learning.utils import Ansatz
from QuICT.algorithm.quantum_machine_learning.utils.gate_tensor import *
from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.simulation.state_vector import ConstantStateVectorSimulator


"""Google Tensorflow Quantum https://arxiv.org/abs/1802.06002
"""


class QuantumNet(nn.Module):
    def __init__(
        self, layers=["XX", "ZZ"], device=torch.device("cuda:0"),
    ):
        super.__init__(self)
        self.layers = layers
        self.device = device
        self.data_qubits = self.resize[0] * self.resize[1]
        self.n_qubits = self.data_qubits + 1
        self.pqc = QNNLayer(
            list(range(self.data_qubits)), self.data_qubits, device=self.device
        )
        self._define_params()

    def forward(self, x):
        data_ansatz = self._qubit_encoding(x)
        model_ansatz = self._construct_ansatz()
        ansatz = data_ansatz + model_ansatz
        _, prob = ansatz.forward()
        assert prob is not None, "There is no Measure Gate on the readout qubit."
        return prob

    def _define_params(self):
        """Define the network parameters to be trained."""
        self.params = nn.Parameter(
            torch.rand(len(self.layers), self.data_qubits, device=self.device),
            requires_grad=True,
        )

    def _qubit_encoding(self, img):
        img = img.flatten()
        data_ansatz = Ansatz(self.data_qubits, device=self.device)
        for i in range(img.shape[0]):
            if img[i]:
                data_ansatz.add_gate(X_tensor, i)
        return data_ansatz

    def _construct_ansatz(self):
        model_ansatz = Ansatz(self.n_qubits, device=self.device)
        model_ansatz.add_gate(X_tensor, self.data_qubits)
        model_ansatz.add_gate(H_tensor, self.data_qubits)
        layer = self.pqc.ansatz_layer(self.layers, self.params)
        model_ansatz += layer
        model_ansatz.add_gate(H_tensor, self.data_qubits)
        model_ansatz.add_gate(Measure_tensor, self.data_qubits)

        return model_ansatz


class ClassicalNet(nn.Module):
    def __init__(self):
        super.__init__(self)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

