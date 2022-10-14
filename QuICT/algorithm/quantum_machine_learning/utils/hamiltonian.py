import copy

import numpy as np
import torch

from QuICT.algorithm.quantum_machine_learning.utils import Ansatz
from QuICT.algorithm.quantum_machine_learning.utils.gate_tensor import *
from QuICT.core import Circuit
from QuICT.core.gate import *


class Hamiltonian:
    @property
    def coefficients(self):
        return self._coefficients

    @property
    def pauli_gates(self):
        return self._pauli_gates

    @property
    def qubit_indexes(self):
        return self._qubit_indexes

    @property
    def h_matrix(self):
        return self._h_matrix

    def __init__(self, pauli_str: list):
        self._pauli_str = pauli_str
        self._remove_I()
        self._coefficients = []
        self._pauli_gates = []
        self._qubit_indexes = []
        self._pauli_str_validation()

    def __getitem__(self, indexes):
        pauli_str = []
        if isinstance(indexes, int):
            indexes = [indexes]
        for idx in indexes:
            pauli_str.append(self._pauli_str[idx])
        return Hamiltonian(pauli_str)

    def __add__(self, other):
        return Hamiltonian(self._pauli_str + other._pauli_str)

    def __sub__(self, other):
        return self.__add__(other.__mul__(-1))

    def __mul__(self, other: float):
        new_pauli_str = copy.deepcopy(self.pauli_str)
        for pauli_operator in new_pauli_str:
            pauli_operator[0] *= other
        return Hamiltonian(new_pauli_str)

    def _remove_I(self):
        new_pauli_str = []
        for pauli_operator in self._pauli_str:
            for pauli_gate in pauli_operator[1:][::-1]:
                if "I" in pauli_gate:
                    pauli_operator.remove(pauli_gate)
            new_pauli_str.append(pauli_operator)

        self._pauli_str = new_pauli_str

    def get_hamiton_matrix(self, n_qubits):
        hamiton_matrix = np.zeros((1 << n_qubits, 1 << n_qubits), dtype=np.complex128)
        hamiton_circuits = self.construct_hamiton_circuit(n_qubits)
        for coeff, circuit in zip(self._coefficients, hamiton_circuits):
            hamiton_matrix += coeff * circuit.matrix()

        return hamiton_matrix

    def construct_hamiton_circuit(self, n_qubits):
        hamiton_circuits = []
        gate_dict = {"X": X, "Y": Y, "Z": Z}
        for qubit_index, pauli_gate in zip(self._qubit_indexes, self._pauli_gates):
            circuit = Circuit(n_qubits)
            for qid, gate in zip(qubit_index, pauli_gate):
                assert gate in gate_dict.keys(), "Invalid Pauli gate."
                gate_dict[gate] | circuit(qid)
            hamiton_circuits.append(circuit)
        return hamiton_circuits

    def construct_hamiton_ansatz(self, n_qubits, device=torch.device("cuda:0")):
        hamiton_ansatz = []
        gate_dict = {"X": X_tensor, "Y": Y_tensor, "Z": Z_tensor}
        for qubit_index, pauli_gate in zip(self._qubit_indexes, self._pauli_gates):
            ansatz = Ansatz(n_qubits, device=device)
            for qid, gate in zip(qubit_index, pauli_gate):
                assert gate in gate_dict.keys(), "Invalid Pauli gate."
                ansatz.add_gate(gate_dict[gate], qid)
            hamiton_ansatz.append(ansatz)
        return hamiton_ansatz

    def _pauli_str_validation(self):
        for pauli_operator in self._pauli_str:
            self._pauli_operator_validation(pauli_operator)

    def _pauli_operator_validation(self, pauli_operator):
        assert isinstance(pauli_operator[0], int) or isinstance(
            pauli_operator[0], float
        ), "The coefficient of a Pauli operator must be integer or float."
        self._coefficients.append(float(pauli_operator[0]))

        assert len(pauli_operator) == len(
            set(pauli_operator)
        ), "The qubit indice of the same Pauli Gate cannot be the same."

        indexes = []
        pauli_gates = ""
        for pauli_gate in pauli_operator[1:]:
            pauli_gate = pauli_gate.upper()
            assert (
                pauli_gate[0] in ["X", "Y", "Z", "I"] and pauli_gate[1:].isdigit()
            ), "The Pauli gate should be in the format of gate + index。 e.g. Z0, I5, Y3."
            pauli_gates += pauli_gate[0]
            indexes.append(int(pauli_gate[1:]))
        self._qubit_indexes.append(indexes)
        self._pauli_gates.append(pauli_gates)
