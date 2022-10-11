import tqdm
import torch

# import torch.utils.tensorboard
from QuICT.algorithm.quantum_machine_learning.VQA.model.QAOANet import QAOANet
from QuICT.algorithm.quantum_machine_learning.utils.hamiltonian import Hamiltonian


class QAOA:
    def __init__(
        self,
        n_qubits: int,
        p: int,
        hamiltonian: Hamiltonian,
        loss_func=None,
        device=torch.device("cuda:0"),
    ):
        self.n_qubits = n_qubits
        self.loss_func = loss_func
        self.net = QAOANet(n_qubits, p, hamiltonian, device).to(device)

    def run(self, optimizer, lr, max_iters):
        optim = optimizer([dict(params=self.net.parameters(), lr=lr)])
        self.net.train()
        loader = tqdm.trange(max_iters, desc="training", leave=False)
        for it in loader:
            optim.zero_grad()
            state = self.net()
            loss = (
                self.net.loss_func(state)
                if self.loss_func is None
                else self.loss_func(state)
            )
            loss.backward()
            optim.step()
            print(loss.item())
        return state
