import torch
import numpy as np
from QuICT.algorithm.quantum_machine_learning.utils.hamiltonian import Hamiltonian
from QuICT.algorithm.quantum_machine_learning.utils.ansatz import Ansatz
from QuICT.simulation.state_vector import ConstantStateVectorSimulator


class VQANet(torch.nn.Module):
    def __init__(
        self,
        n_qubits: int,
        p: int,
        hamiltonian: Hamiltonian,
        device=torch.device("cuda:0"),
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.p = p
        self.hamiltonian = hamiltonian
        self.device = device
        self.define_network()

    def define_network(self):
        raise NotImplementedError

    def loss_func(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(self.device)
        assert state.shape[0] == 1 << self.n_qubits

        ansatz_list = self.hamiltonian.construct_hamiton_ansatz(
            self.n_qubits, self.device
        )
        coefficients = self.hamiltonian.coefficients
        state_vector = torch.zeros(1 << self.n_qubits, dtype=torch.complex128).to(
            self.device
        )
        for coeff, ansatz in zip(coefficients, ansatz_list):
            sv = ansatz.forward(state)
            state_vector += coeff * sv

        loss = torch.sum(state * state_vector).real
        return loss


if __name__ == "__main__":
    import random

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
        state /= sum(abs(state))
        state = abs(state) ** 0.5
        state = state.astype(np.complex128)
        return state

    def seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    n_qubits = 5
    pauli_str = random_pauli_str(2, n_qubits)
    print(pauli_str)
    h = Hamiltonian(pauli_str)
    state = random_state(n_qubits)
    net = VQANet(hamiltonian=h, p=1, n_qubits=n_qubits)
    loss = net.loss_func(state)
    print(loss)
