import torch
import numpy as np

from QuICT.algorithm.quantum_machine_learning.utils import Ansatz
from QuICT.algorithm.quantum_machine_learning.utils.gate_tensor import *
from QuICT.core import Circuit
from QuICT.core.gate import *


class QNNLayer(torch.nn.Module):
    def __init__(self, data_qubits, result_qubit, device=torch.device("cuda:0")):
        super().__init__()
        self.n_qubits = len(data_qubits) + 1
        assert (
            result_qubit < self.n_qubits and result_qubit not in data_qubits
        ), "Wrong result qubit."
        self.data_qubits = data_qubits
        self.result_qubit = result_qubit
        self.device = device

    def ansatz_layer(self, two_qubit_gates, params):
        if not isinstance(two_qubit_gates, list):
            two_qubit_gates = [two_qubit_gates]
        n_layers = len(two_qubit_gates)
        assert (
            params.shape[0] == n_layers and params.shape[1] == self.n_qubits - 1
        ), "The shape of the parameters should be [n_layers, n_data_qubits]."

        gate_dict = {
            "XX": Rxx_tensor,
            "YY": Ryy_tensor,
            "ZZ": Rzz_tensor,
            "ZX": Rzx_tensor,
        }
        ansatz = Ansatz(self.n_qubits, device=self.device)
        for l, gate in zip(range(n_layers), two_qubit_gates):
            assert (
                gate in gate_dict.keys()
            ), "Invalid Two Qubit Gate. Should be XX, YY, ZZ or ZX."

            for i in range(self.n_qubits - 1):
                ansatz.add_gate(
                    gate_dict[gate](params[l][i]),
                    [self.data_qubits[i], self.result_qubit],
                )
        return ansatz

    def circuit_layer(self, two_qubit_gates, params):
        if not isinstance(two_qubit_gates, list):
            two_qubit_gates = [two_qubit_gates]
        n_layers = len(two_qubit_gates)
        assert (
            params.shape[0] == n_layers and params.shape[1] == self.n_qubits - 1
        ), "The shape of the parameters should be [n_layers, n_data_qubits]."

        gate_dict = {
            "XX": Rxx,
            "YY": Ryy,
            "ZZ": Rzz,
            "ZX": Rzx,
        }
        ansatz = Ansatz(self.n_qubits, device=self.device)
        for l, gate in zip(range(n_layers), two_qubit_gates):
            assert (
                gate in gate_dict.keys()
            ), "Invalid Two Qubit Gate. Should be XX, YY, ZZ or ZX."

            for i in range(self.n_qubits - 1):
                ansatz.add_gate(
                    gate_dict[gate](params[l][i]),
                    [self.data_qubits[i], self.result_qubit],
                )
        return ansatz

