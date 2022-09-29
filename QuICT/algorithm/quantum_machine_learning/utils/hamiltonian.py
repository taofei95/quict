from typing import Dict, List, Union
import numpy as np
import copy
import torch

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.simulation.state_vector import ConstantStateVectorSimulator
from QuICT.ops.linalg import gpu_calculator
from QuICT.algorithm.quantum_machine_learning.utils.ansatz import Ansatz


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

    def get_hamiton_matrix(self, n_qubits):
        hamiton_matrix = np.zeros((1 << n_qubits, 1 << n_qubits), dtype=np.complex128)
        for coeff, qubit_index, pauli_gate in zip(
            self._coefficients, self._qubit_indexes, self._pauli_gates
        ):
            matrix = np.array([1], dtype=np.complex128)
            num = 0
            for i in range(n_qubits):
                if i not in qubit_index:
                    matrix = np.kron(matrix, np.eye(2, dtype=np.complex128))
                    num += 1
                else:
                    gate = pauli_gate[i - num]
                    if gate == "X":
                        matrix = np.kron(matrix, X.matrix)
                    elif gate == "Y":
                        matrix = np.kron(matrix, Y.matrix)
                    elif gate == "Z":
                        matrix = np.kron(matrix, Z.matrix)
                    elif gate == "I":
                        matrix = np.kron(matrix, np.eye(2, dtype=np.complex128))
                    else:
                        raise ValueError("Invalid Pauli gate")

            hamiton_matrix += coeff * matrix
        return hamiton_matrix

    def construct_hamiton_circuit(self, n_qubits):
        hamiton_circuits = []
        for qubit_index, pauli_gate in zip(self._qubit_indexes, self._pauli_gates):
            circuit = Circuit(n_qubits)
            for qid, gate in zip(qubit_index, pauli_gate):
                if gate == "X":
                    X | circuit(qid)
                elif gate == "Y":
                    Y | circuit(qid)
                elif gate == "Z":
                    Z | circuit(qid)
                elif gate == "I":
                    continue
                else:
                    raise ValueError("Invalid Pauli gate.")
            hamiton_circuits.append(circuit)
        return hamiton_circuits
    
    def construct_hamiton_ansatz(self, n_qubits, device=torch.device("cuda:0")):
        hamiton_ansatz = []
        for qubit_index, pauli_gate in zip(self._qubit_indexes, self._pauli_gates):
            circuit = Circuit(n_qubits)
            for qid, gate in zip(qubit_index, pauli_gate):
                if gate == "X":
                    X | circuit(qid)
                elif gate == "Y":
                    Y | circuit(qid)
                elif gate == "Z":
                    Z | circuit(qid)
                elif gate == "I":
                    continue
                else:
                    raise ValueError("Invalid Pauli gate.")
            ansatz = Ansatz(circuit, device)
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


def main():
    state = np.array([np.sqrt(3) / 3, 1 / 2, 1 / 3, np.sqrt(11) / 6])
    state = state.reshape(-1, 1)
    h = Hamiltonian([[0.2, "Z0", "I1"], [1, "X1"]])
    # pauli_str = [
    #     [-1.0, "z0", "z1"],
    #     [-1.0, "z1", "z2"],
    #     [-1.0, "z2", "z3"],
    #     [-1.0, "z3", "z0"],
    # ]
    # h = Hamiltonian(pauli_str)
    hamiton_matrix = h.get_hamiton_matrix(2)
    print(hamiton_matrix.real)
    print(state.T @ hamiton_matrix.real @ state)
    # print(h._pauli_gates)
    # print(h._qubit_indexes)
    # print(h._h_matrix)
    # circuit = h.construct_hamiton_circuit(2)
    # simulator = ConstantStateVectorSimulator()
    # sv1 = simulator.run(circuit[0], state)
    # sv2 = simulator.run(circuit[1], state)
    # print(state.T@sv1.get().real)
    # print(sum(state*sv1.get().real))
    # print((0.2*sv1.real + sv2.real))


if __name__ == "__main__":
    main()

