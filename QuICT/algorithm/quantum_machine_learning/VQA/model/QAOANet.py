from fileinput import filename
import torch
import torch.nn
import re

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.algorithm.quantum_machine_learning.utils.gate_tensor import *
from QuICT.algorithm.quantum_machine_learning.utils.hamiltonian import Hamiltonian
from QuICT.algorithm.quantum_machine_learning.utils.ansatz import Ansatz
from QuICT.algorithm.quantum_machine_learning.VQA.model.VQANet import VQANet


class QAOANet(VQANet):
    def __init__(
        self,
        n_qubits: int,
        p: int,
        hamiltonian: Hamiltonian,
        device=torch.device("cuda:0"),
    ):
        super().__init__(n_qubits, p, hamiltonian, device)

    def define_network(self):
        self.beta = torch.nn.Parameter(
            torch.rand(self.p, device=self.device), requires_grad=True
        )
        self.gamma = torch.nn.Parameter(
            torch.rand(self.p, device=self.device), requires_grad=True
        )

    def forward(self, state=None):
        ansatz = self.construct_ansatz()
        state = ansatz.forward(state)
        return state

    def _construct_U_gamma_layer(self, gamma):
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
                        ansatz.add_gate(H_tensor, qids[i])
                    elif gates[i] == "Y":
                        ansatz.add_gate(Hy_tensor, qids[i])
                    elif gates[i] == "Z":
                        continue
                    else:
                        raise ValueError("Invalid Pauli gate.")
                ansatz = ansatz + self._Rnz_ansatz(2 * coeff * gamma, qids)
                for i in range(len(qids)):
                    if gates[i] == "X":
                        ansatz.add_gate(H_tensor, qids[i])
                    elif gates[i] == "Y":
                        ansatz.add_gate(Hy_tensor, qids[i])
                    elif gates[i] == "Z":
                        continue
                    else:
                        raise ValueError("Invalid Pauli gate.")
            # Only Rx, Ry, Rz
            elif len(qids) == 1:
                if gates[0] == "X":
                    ansatz.add_gate(Rx_tensor(2 * coeff * gamma), qids)
                elif gates[0] == "Y":
                    ansatz.add_gate(Ry_tensor(2 * coeff * gamma), qids)
                elif gates[0] == "Z":
                    ansatz.add_gate(Rz_tensor(2 * coeff * gamma), qids)
                else:
                    raise ValueError("Invalid Pauli gate.")
            # Only coeff
            else:
                ansatz.add_gate(RI_tensor(-coeff))

        return ansatz

    def _Rnz_ansatz(self, gamma, tar_idx: Union[int, list]):
        ansatz = Ansatz(n_qubits=self.n_qubits, device=self.device)
        if isinstance(tar_idx, int):
            ansatz.add_gate(Rz_tensor(gamma), tar_idx)
        else:
            # Add CNOT gates
            for i in range(len(tar_idx) - 1):
                ansatz.add_gate(CX_tensor, tar_idx[i : i + 2])
            # Add RZ gate
            ansatz.add_gate(Rz_tensor(gamma), tar_idx[-1])
            # Add CNOT gates
            for i in range(len(tar_idx) - 2, -1, -1):
                ansatz.add_gate(CX_tensor, tar_idx[i : i + 2])
        return ansatz

    def construct_ansatz(self):
        ansatz = Ansatz(self.n_qubits, device=self.device)
        # initialize state vector
        ansatz.add_gate(H_tensor)

        for p in range(self.p):
            # construct U_gamma
            U_gamma = self._construct_U_gamma_layer(self.gamma[p])
            ansatz = ansatz + U_gamma

            # construct U_beta
            U_beta = Rx_tensor(2 * self.beta[p])
            ansatz.add_gate(U_beta)

        return ansatz

    def _construct_U_gamma_circuit(self, gamma):
        circuit = Circuit(self.n_qubits)
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
                        H | circuit(qids[i])
                    elif gates[i] == "Y":
                        Hy | circuit(qids[i])
                    elif gates[i] == "Z":
                        continue
                    else:
                        raise ValueError("Invalid Pauli gate.")
                Rnz_circuit = self._Rnz_circuit(2 * coeff * gamma, qids)
                circuit.extend(Rnz_circuit.gates)

                for i in range(len(qids)):
                    if gates[i] == "X":
                        H | circuit(qids[i])
                    elif gates[i] == "Y":
                        Hy | circuit(qids[i])
                    elif gates[i] == "Z":
                        continue
                    else:
                        raise ValueError("Invalid Pauli gate.")
            # Only Rx, Ry, Rz
            elif len(qids) == 1:
                if gates[0] == "X":
                    Rx(2 * coeff * gamma) | circuit(qids[0])
                elif gates[0] == "Y":
                    Ry(2 * coeff * gamma) | circuit(qids[0])
                elif gates[0] == "Z":
                    Rz(2 * coeff * gamma) | circuit(qids[0])
                else:
                    raise ValueError("Invalid Pauli gate.")
            # Only coeff
            else:
                RI | circuit

        return circuit

    def _Rnz_circuit(self, gamma, tar_idx: Union[int, list]):
        circuit = Circuit(self.n_qubits)
        if isinstance(tar_idx, int):
            Rz(gamma) | circuit(tar_idx)
        else:
            # Add CNOT gates
            for i in range(len(tar_idx) - 1):
                CX | circuit(tar_idx[i : i + 2])
            # Add RZ gate
            Rz(gamma) | circuit(tar_idx[-1])
            # Add CNOT gates
            for i in range(len(tar_idx) - 2, -1, -1):
                CX | circuit(tar_idx[i : i + 2])
        return circuit

    def construct_circuit(self):
        gamma = self.gamma.cpu().detach().numpy()
        beta = self.beta.cpu().detach().numpy()
        circuit = Circuit(self.n_qubits)
        # initialize state vector
        H | circuit

        for p in range(self.p):
            # construct U_gamma
            U_gamma = self._construct_U_gamma_circuit(gamma[p])
            circuit.extend(U_gamma.gates)

            # construct U_beta
            U_beta = Rx(2 * beta[p])
            U_beta | circuit

        return circuit
