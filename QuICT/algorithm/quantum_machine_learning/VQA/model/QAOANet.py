import torch.nn
import numpy as np
import random
import time
import re

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.algorithm.quantum_machine_learning.utils.hamiltonian import Hamiltonian
from QuICT.algorithm.quantum_machine_learning.utils.ansatz import Ansatz
from QuICT.algorithm.quantum_machine_learning.VQA.model.VQANet import VQANet
from QuICT.simulation.state_vector import ConstantStateVectorSimulator
from QuICT.qcda.synthesis.unitary_decomposition import UnitaryDecomposition


class QAOANet(VQANet):
    def __init__(
        self,
        n_qubits: int,
        p: int,
        hamiltonian: Hamiltonian,
        device=torch.device("cuda:0"),
    ):
        super().__init__(n_qubits, p, hamiltonian, device)
        self.define_network()

    def define_network(self):
        self.beta = torch.nn.Parameter(
            torch.rand(self.p, device=self.device), requires_grad=True
        )
        self.gamma = torch.nn.Parameter(
            torch.rand(self.p, device=self.device), requires_grad=True
        )

    def forward(self, state=None):
        ansatz = self.construct_ansatz(self.gamma, self.beta)
        state = ansatz.forward(state)
        return state

    def construct_U_gamma_layer(self, gamma):
        ansatz = Ansatz(n_qubits=self.n_qubits, device=self.device)
        for coeff, qids, gates in zip(
            self.hamiltonian._coefficients,
            self.hamiltonian._qubit_indexes,
            self.hamiltonian._pauli_gates,
        ):
            # Remove I from pauli_gates and remove corresponding qid from qubit_indexes
            findI = [i.start() for i in re.finditer("I", gates)]
            gates = gates.replace("I", "")
            for i in findI:
                qids.pop(i)

            # Mapping e.g. Rxyz
            if len(qids) > 1:
                for i in range(len(qids)):
                    if gates[i] == "X":
                        ansatz.add_gate(H, qids[i])
                    elif gates[i] == "Y":
                        ansatz.add_gate(Hy, qids[i])
                    elif gates[i] == "Z":
                        continue
                    else:
                        raise ValueError("Invalid Pauli gate.")
                ansatz = ansatz + self._Rnz_ansatz(2 * coeff * gamma, qids)
                for i in range(len(qids)):
                    if gates[i] == "X":
                        ansatz.add_gate(H, qids[i])
                    elif gates[i] == "Y":
                        ansatz.add_gate(Hy, qids[i])
                    elif gates[i] == "Z":
                        continue
                    else:
                        raise ValueError("Invalid Pauli gate.")
            # Only Rx, Ry, Rz
            elif len(qids) == 1:
                if gates[0] == "X":
                    ansatz.add_gate(Rx(2 * coeff * gamma), qids)
                elif gates[0] == "Y":
                    ansatz.add_gate(Ry(2 * coeff * gamma), qids)
                elif gates[0] == "Z":
                    ansatz.add_gate(Rz(2 * coeff * gamma), qids)
                else:
                    raise ValueError("Invalid Pauli gate.")
            # Only coeff
            else:
                RI = np.array(
                    [[np.exp(-coeff * 1j), 0], [0, np.exp(-coeff * 1j)]],
                    dtype=np.complex128,
                )
                RI = Unitary(RI)
                ansatz.add_gate(RI)

        return ansatz

    def _Rnz_ansatz(self, gamma, tar_idx: Union[int, list]):
        ansatz = Ansatz(n_qubits=self.n_qubits, device=self.device)
        if isinstance(tar_idx, int):
            ansatz.add_gate(Rz(gamma), tar_idx)
        else:
            # Add CNOT gates
            for i in range(len(tar_idx) - 1):
                ansatz.add_gate(CX, tar_idx[i : i + 2])
            # Add RZ gate
            ansatz.add_gate(Rz(gamma), tar_idx[-1])
            # Add CNOT gates
            for i in range(len(tar_idx) - 2, -1, -1):
                ansatz.add_gate(CX, tar_idx[i : i + 2])
        return ansatz

    def construct_ansatz(self, gamma, beta):
        start = time.time()
        ansatz = Ansatz(self.n_qubits, device=self.device)
        # initialize state vector
        ansatz.add_gate(H)

        for p in range(self.p):
            # construct U_gamma
            U_gamma = self.construct_U_gamma_layer(gamma[p])
            ansatz = ansatz + U_gamma

            # construct U_beta
            U_beta = Rx(2 * beta[p])
            ansatz.add_gate(U_beta)

        end = time.time() - start
        return ansatz


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

    n_qubits = 2
    pauli_str = random_pauli_str(2, n_qubits)
    print(pauli_str)
    h = Hamiltonian(pauli_str)
    state = random_state(n_qubits)
    # h = Hamiltonian([[0.2, "Z0", "I1"], [1, "X1"]])
    net = QAOANet(hamiltonian=h, p=1, n_qubits=n_qubits)
    # state = np.array([np.sqrt(3) / 3, 1 / 2, 1 / 3, np.sqrt(11) / 6])
