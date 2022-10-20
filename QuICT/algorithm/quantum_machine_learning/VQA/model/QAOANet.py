import torch
import torch.nn

from QuICT.algorithm.quantum_machine_learning.utils import Ansatz, Hamiltonian
from QuICT.algorithm.quantum_machine_learning.utils.gate_tensor import *
from QuICT.algorithm.quantum_machine_learning.VQA.model import VQENet
from QuICT.core import Circuit
from QuICT.core.gate import *


class QAOANet(VQENet):
    """The network used by QAOA.

    The QAOANet implementation directly extends VQENet and inherits its optimization structure.
    However, unlike VQE, which can be configured with arbitrary ansatzes, QAOA uses its
    own fine-tuned ansatz.
    """

    def __init__(
        self,
        n_qubits: int,
        p: int,
        hamiltonian: Hamiltonian,
        device=torch.device("cuda:0"),
    ):
        """Initialize a QAOANet instance.

        Args:
            n_qubits (int): The number of qubits.
            p (int): The number of layers of the network.
            hamiltonian (Hamiltonian): The hamiltonian for a specific task.
            device (torch.device, optional): The device to which the QAOANet is assigned.
                Defaults to torch.device("cuda:0").
        """
        super().__init__(n_qubits, p, hamiltonian, device)

    def define_network(self):
        """Define the network parameters to be trained."""
        self.beta = torch.nn.Parameter(
            torch.rand(self.p, device=self.device), requires_grad=True
        )
        self.gamma = torch.nn.Parameter(
            torch.rand(self.p, device=self.device), requires_grad=True
        )

    def forward(self, state=None):
        """The forward propagation process of QAOANet.

        Args:
            state (np.array/torch.Tensor, optional): The input state vector.
                Defaults to None, which means the initial state |0>.

        Returns:
            torch.Tensor: The output state vector.
        """
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
            # Mapping e.g. Rxyz
            if len(qids) > 1:
                gate_dict = {"X": H_tensor, "Y": Hy_tensor}
                for i in range(len(qids)):
                    if gates[i] != "Z":
                        assert gates[i] in gate_dict.keys(), "Invalid Pauli gate."
                        ansatz.add_gate(gate_dict[gates[i]], qids[i])
                ansatz = ansatz + self._Rnz_ansatz(2 * coeff * gamma, qids)
                for i in range(len(qids)):
                    if gates[i] != "Z":
                        assert gates[i] in gate_dict.keys(), "Invalid Pauli gate."
                        ansatz.add_gate(gate_dict[gates[i]], qids[i])

            # Only Rx, Ry, Rz
            elif len(qids) == 1:
                gate_dict = {
                    "X": Rx_tensor(2 * coeff * gamma),
                    "Y": Ry_tensor(2 * coeff * gamma),
                    "Z": Rz_tensor(2 * coeff * gamma),
                }
                assert gates[0] in gate_dict.keys(), "Invalid Pauli gate."
                ansatz.add_gate(gate_dict[gates[0]], qids[0])

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
        """Build QAOA ansatz with optimizable parameters.

        Returns:
            Ansatz: The QAOA ansatz.
        """
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
            # Mapping e.g. Rxyz
            if len(qids) > 1:
                gate_dict = {"X": H, "Y": Hy}
                for i in range(len(qids)):
                    if gates[i] != "Z":
                        assert gates[i] in gate_dict.keys(), "Invalid Pauli gate."
                        gate_dict[gates[i]] | circuit(qids[i])
                Rnz_circuit = self._Rnz_circuit(2 * coeff * gamma, qids)
                circuit.extend(Rnz_circuit.gates)
                for i in range(len(qids)):
                    if gates[i] != "Z":
                        assert gates[i] in gate_dict.keys(), "Invalid Pauli gate."
                        gate_dict[gates[i]] | circuit(qids[i])

            # Only Rx, Ry, Rz
            elif len(qids) == 1:
                gate_dict = {
                    "X": Rx(2 * coeff * gamma),
                    "Y": Ry(2 * coeff * gamma),
                    "Z": Rz(2 * coeff * gamma),
                }
                assert gates[0] in gate_dict.keys(), "Invalid Pauli gate."
                gate_dict[gates[0]] | circuit(qids[0])

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
        """Build QAOA circuit with optimizable parameters.

        Returns:
            Circuit: The QAOA circuit.
        """
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

    def loss_func(self, state):
        """The loss function for QAOA, as opposed to VQE, which aims to maximize the expectation of H.

        Args:
            state (torch.Tensor): The state vector.

        Returns:
            torch.Tensor: Loss, which is equal to the negative expectation of H.
        """
        return -super().loss_func(state)
