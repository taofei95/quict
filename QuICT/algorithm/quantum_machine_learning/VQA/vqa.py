import numpy as np
import torch
from torch.optim import Optimizer
import time

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.simulation.state_vector import ConstantStateVectorSimulator
from QuICT.algorithm.quantum_machine_learning.utils.hamiltonian import Hamiltonian
from QuICT.ops.linalg import gpu_calculator

# from QuICT.algorithm.quantum_machine_learning.utils.ansatz import Ansatz


class VQA:
    # def __init__(
    #     self, ansatz: Ansatz, hamiltonian: Hamiltonian, init_params: np.ndarray,
    # ):
    #     self._ansatz = ansatz
    #     self._hamiltonian = hamiltonian
    #     self._init_params = init_params

    def __init__(
        self, hamiltonian: Hamiltonian, simulator=ConstantStateVectorSimulator()
    ):
        self._hamiltonian = hamiltonian
        self._simulator = simulator

    def cal_expect(self, state):
        n_qubits = int(np.log2(len(state)))
        circuits = self._hamiltonian.construct_hamiton_circuit(n_qubits)
        coefficients = self._hamiltonian.coefficients
        state_vector = np.zeros(len(state))
        for coeff, circuit in zip(coefficients, circuits):
            sv = self._simulator.run(circuit, state)
            state_vector += coeff * sv.get().real
        expect = sum(state * state_vector)
        return expect

    def run(self):
        raise NotImplementedError


if __name__ == "__main__":

    def random_pauli_str(n_items, n_qubits):
        pauli_str = []
        coeffs = np.random.rand(n_items)
        for i in range(n_items):
            pauli = [coeffs[i]]
            for qid in range(n_qubits):
                flag = np.random.randint(0, 5)
                if flag == 0:
                    pauli.append("X" + str(qid))
                elif flag == 1:
                    pauli.append("Y" + str(qid))
                elif flag == 2:
                    pauli.append("Z" + str(qid))
                elif flag == 3:
                    pauli.append("I" + str(qid))
                elif flag == 4:
                    continue
            pauli_str.append(pauli)
        return pauli_str

    def random_state(n_qubits):
        state = np.random.randn(1 << n_qubits)
        state /= sum(state)
        state = abs(state) ** 0.5
        print(sum(state * state))
        return state

    n_qubits = 1
    pauli_str = random_pauli_str(19, n_qubits)
    # print(pauli_str)
    h = Hamiltonian(pauli_str)
    state = random_state(n_qubits)
    # print(state)
    # h = Hamiltonian([[0.2, "Z0", "I1"], [1, "X1"]])
    vqa = VQA(hamiltonian=h)
    # state = np.array([np.sqrt(3) / 3, 1 / 2, 1 / 3, np.sqrt(11) / 6])
    loss = vqa.cal_expect(state)
    print(loss)

