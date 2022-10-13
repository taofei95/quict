from pyexpat import model
import tqdm
import torch
import os
import shutil
import time
from typing import Union

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
        self.model_path = None
        self.optim = None

    def save_checkpoint(self, it, latest=False):
        os.makedirs(self.model_path, exist_ok=True)
        checkpoint = dict(
            iter=it, graph=self.net.state_dict(), optimizer=self.optim.state_dict()
        )
        torch.save(checkpoint, "{0}/model.ckpt".format(self.model_path))
        if not latest:
            shutil.copy(
                "{0}/model.ckpt".format(self.model_path),
                "{0}/{1}.ckpt".format(self.model_path, it),
            )

    def restore_checkpoint(self, resume):
        assert resume and self.model_path is not None
        try:
            model_name = (
                "{0}/model.ckpt".format(self.model_path)
                if resume is True
                else "{0}/{1}.ckpt".format(self.model_path, resume)
            )
        except:
            raise Exception("Cannot find the model.")

        checkpoint = torch.load(model_name, map_location=self.net.device)
        try:
            self.net.load_state_dict(checkpoint["graph"])
            if self.optim:
                self.optim.load_state_dict(checkpoint["optimizer"])
        except:
            raise Exception("Cannot load the model correctly.")

        it = checkpoint["iter"]
        if resume is not True:
            assert resume == it

        return it

    def train(
        self,
        optimizer: torch.optim.Optimizer,
        lr: float,
        max_iters: int,
        save_model=True,
        model_path=None,
        resume: Union[bool, int] = False,
    ):
        self.model_path = model_path
        self.optim = optimizer([dict(params=self.net.parameters(), lr=lr)])
        # Restore checkpoint
        curr_it = (
            self.restore_checkpoint(resume=resume) if resume and self.model_path else 0
        )

        # Start training
        self.net.train()
        assert curr_it < max_iters
        loader = tqdm.trange(curr_it, max_iters, desc="Training", leave=False)
        for it in loader:
            self.optim.zero_grad()
            state = self.net()
            loss = (
                self.net.loss_func(state)
                if self.loss_func is None
                else self.loss_func(state)
            )
            loss.backward()
            self.optim.step()
            loader.set_postfix(loss=loss.item())

            # Save checkpoint
            if save_model and ((it + 1) % 10 == 0 or (it + 1) == max_iters):
                if self.model_path is None:
                    now = time.strftime(
                        "%Y-%m-%d-%H_%M_%S", time.localtime(time.time())
                    )
                    self.model_path = "QAOA_model_" + now
                self.save_checkpoint(it + 1, latest=(it + 1 == max_iters))

        return state

    def test(self, model_path):
        assert model_path is not None and model_path != ""
        self.model_path = model_path
        # Restore checkpoint
        self.restore_checkpoint(resume=True)
        self.net.eval()
        state = self.net()
        return state


if __name__ == "__main__":
    import numpy as np
    import random
    import torch
    import matplotlib.pyplot as plt
    from QuICT.simulation.state_vector import ConstantStateVectorSimulator
    from QuICT.algorithm.quantum_machine_learning.utils.hamiltonian import Hamiltonian

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

    def seed(seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def draw_prob(prob, shots, model_path=None):
        plt.figure()
        plt.xlabel("Qubit States")
        plt.ylabel("Probabilities")
        plt.bar(range(len(prob)), np.array(prob) / shots)
        if model_path is not None:
            plt.savefig(model_path + "/Probabilities.jpg")
        plt.show()

    seed(0)
    n_items = 3
    n_qubits = 5
    pauli_list = random_pauli_str(n_items, n_qubits)
    print(pauli_list)
    hamiltonian = Hamiltonian(pauli_list)
    qaoa = QAOA(n_qubits, p=10, hamiltonian=hamiltonian)
    state = qaoa.train(
        optimizer=torch.optim.Adam, lr=0.1, max_iters=100, save_model=True
    )
    # state = qaoa.test(model_path="QAOA_model_2022-10-13-15_17_48") 
    expect = -qaoa.net.loss_func(state)
    print("Expect: ", expect)

    hamiltonian_matrix = hamiltonian.get_hamiton_matrix(n_qubits)
    eigens = np.linalg.eig(hamiltonian_matrix)
    max_eigen = np.real(eigens[0][np.argmax(eigens[0])])
    print("Max Eigen: ", max_eigen)

    # circuit = qaoa.net.construct_circuit()
    # circuit.draw(filename=qaoa.model_path + "/circuit.jpg")
