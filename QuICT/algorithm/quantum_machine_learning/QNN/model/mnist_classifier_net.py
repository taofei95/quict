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
        self,
        data_qubits,
        layers=["XX", "ZZ"],
        encoding="qubit",
        device=torch.device("cuda:0"),
    ):
        super(QuantumNet, self).__init__()
        assert encoding in ["qubit", "amplitude"]
        self._layers = layers
        self._encoding = encoding
        self._device = device
        self._data_qubits = data_qubits
        self._n_qubits = self._data_qubits + 1
        self._pqc = QNNLayer(
            list(range(self._data_qubits)), self._data_qubits, device=self._device
        )
        self._define_params()

    def forward(self, X):
        Y_pred = torch.zeros([X.shape[0]], device=self._device)
        for i in range(X.shape[0]):
            if self._encoding == "qubit":
                data_ansatz = self._qubit_encoding(X[i])
            else:
                data_ansatz = self._amplitude_encoding(X[i])
            model_ansatz = self._construct_ansatz()
            ansatz = data_ansatz + model_ansatz
            _, prob = ansatz.forward()
            assert prob is not None, "There is no Measure Gate on the readout qubit."
            Y_pred[i] = prob[1]
            # circuit = self._qubit_encoding_circuit(X[i])
            # model_circuit = self._construct_circuit()
            # circuit.extend(model_circuit.gates)
            # simulator = ConstantStateVectorSimulator()
            # sv = simulator.run(circuit)
            
            
        return Y_pred

    def _define_params(self):
        """Define the network parameters to be trained."""
        self.params = nn.Parameter(
            torch.rand(len(self._layers), self._data_qubits, device=self._device),
            requires_grad=True,
        )

    def _qubit_encoding(self, img):
        img = img.flatten()
        data_ansatz = Ansatz(self._data_qubits, device=self._device)
        for i in range(img.shape[0]):
            if img[i]:
                data_ansatz.add_gate(X_tensor, i)
        return data_ansatz
    
    def _qubit_encoding_circuit(self, img):
        img = img.flatten()
        data_circuit = Circuit(self._n_qubits)
        for i in range(img.shape[0]):
            if img[i]:
                X | data_circuit(i)
        return data_circuit

    def _amplitude_encoding(self, img):
        return

    def _construct_ansatz(self):
        model_ansatz = Ansatz(self._n_qubits, device=self._device)
        model_ansatz.add_gate(X_tensor, self._data_qubits)
        model_ansatz.add_gate(H_tensor, self._data_qubits)
        model_ansatz += self._pqc(self._layers, self.params)
        model_ansatz.add_gate(H_tensor, self._data_qubits)
        model_ansatz.add_gate(Measure_tensor, self._data_qubits)
        return model_ansatz
    
    def _construct_circuit(self):
        model_circuit = Circuit(self._n_qubits)
        X | model_circuit(self._data_qubits)
        H | model_circuit(self._data_qubits)
        sub_circuit = self._pqc.circuit_layer(self._layers, self.params)
        model_circuit.extend(sub_circuit.gates)
        H | model_circuit(self._data_qubits)
        Measure | model_circuit(self._data_qubits)
        return model_circuit


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

