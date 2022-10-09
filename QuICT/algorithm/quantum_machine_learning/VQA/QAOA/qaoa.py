import numpy as np
import os, sys, time
import random
import tqdm
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F

# import torch.utils.tensorboard
from QuICT.algorithm.quantum_machine_learning.VQA.model.QAOANet import QAOANet
from QuICT.algorithm.quantum_machine_learning.utils.hamiltonian import Hamiltonian
from QuICT.simulation.state_vector import ConstantStateVectorSimulator


class QAOA:
    def __init__(
        self, n_qubits, p, hamiltonian: Hamiltonian, device=torch.device("cuda:0")
    ):
        self.n_qubits = n_qubits
        self.net = QAOANet(n_qubits, p, hamiltonian, device).to(device)

    def run(self, optimizer, lr, max_iter):
        optim = optimizer([dict(params=self.net.parameters(), lr=lr)])

        self.net.train()
        loader = tqdm.trange(max_iter, desc="training", leave=False)
        for it in loader:
            optim.zero_grad()
            state = self.net()
            loss = self.net.loss_func(state)
            loss.backward()
            optim.step()
            print(loss.item())


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
        state /= sum(abs(state))
        state = abs(state) ** 0.5
        return state

    def seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    seed(17)
    n_qubits = 2
    pauli_str = random_pauli_str(2, n_qubits)
    print(pauli_str)
    h = Hamiltonian(pauli_str)
    # h = Hamiltonian([[0.2, "Z0", "I1"], [1, "X1"]])
    qaoa = QAOA(n_qubits, 1, h)
    qaoa.run(optimizer=torch.optim.Adam, lr=0.1, max_iter=10)
    # state = np.array([np.sqrt(3) / 3, 1 / 2, 1 / 3, np.sqrt(11) / 6])
