import torch
import numpy as np
from QuICT.algorithm.quantum_machine_learning.utils.hamiltonian import Hamiltonian
from QuICT.algorithm.quantum_machine_learning.utils.ansatz import Ansatz


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

        loss = - torch.sum(state.real * state_vector.real)
        # print(state)
        # print(state_vector)
        # print(state * state_vector)
        
        # hamiton_matrix = self.hamiltonian.get_hamiton_matrix(self.n_qubits)
        # hamiton_matrix = torch.from_numpy(hamiton_matrix).to(self.device)
        # state = state.reshape(1, -1)
        # loss = -torch.mm(state, torch.mm(hamiton_matrix, state.T))[0].real
        
        # print(state)
        # print(torch.mm(hamiton_matrix, state.T))
        
        return loss
