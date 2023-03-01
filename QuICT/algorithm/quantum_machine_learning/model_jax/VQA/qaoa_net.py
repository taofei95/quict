import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random

from QuICT.algorithm.quantum_machine_learning.utils import Ansatz, Hamiltonian
from QuICT.algorithm.quantum_machine_learning.utils.gate_tensor import *
from QuICT.algorithm.quantum_machine_learning.model_jax.VQA import VQENet
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
        key,
        dtype=jnp.complex64,
        device="gpu:0",
    ):
        """Initialize a QAOANet instance.

        Args:
            n_qubits (int): The number of qubits.
            p (int): The number of layers of the network.
            hamiltonian (Hamiltonian): The hamiltonian for a specific task.
            device (torch.device, optional): The device to which the QAOANet is assigned.
                Defaults to torch.device("cuda:0").
        """
        super().__init__(n_qubits, p, hamiltonian, key, dtype, device)

    def define_network(self):
        """Define the network parameters to be trained."""
        self.params = random.normal(self.key, (2, self.p))
        self.params = jax.device_put(self.params, self.device)

    def forward(self, state=None):
        """The forward propagation process of QAOANet.

        Args:
            state (np.array/torch.Tensor, optional): The input state vector.
                Defaults to None, which means the initial state |0>.

        Returns:
            torch.Tensor: The output state vector.
        """
        cirucit = self.construct_cirucit()
        if self.device.type == "cpu":
            raise NotImplementedError
        else:
            state = self.cir_simulator.run(cirucit, state)
        state = jnp.asarray(state)
        state = jax.device_put(state, self.device)
        return state

    def _construct_U_gamma_circuit(self, gamma):
        circuit = Circuit(self.n_qubits)
        for coeff, qids, gates in zip(
            self.hamiltonian._coefficients,
            self.hamiltonian._qubit_indexes,
            self.hamiltonian._pauli_gates,
        ):
            gate_dict = {
                "X": {"mqids": H, "qid": Rx(2 * coeff * gamma)},
                "Y": {"mqids": Hy, "qid": Ry(2 * coeff * gamma)},
                "Z": {"qid": Rz(2 * coeff * gamma)},
            }

            # Mapping e.g. Rxyz
            if len(qids) > 1:
                for i in range(len(qids)):
                    if gates[i] != "Z":
                        gate_dict[gates[i]]["mqids"] | circuit(qids[i])
                Rnz_circuit = self._Rnz_circuit(2 * coeff * gamma, qids)
                circuit.extend(Rnz_circuit.gates)
                for i in range(len(qids)):
                    if gates[i] != "Z":
                        gate_dict[gates[i]]["mqids"] | circuit(qids[i])

            # Only Rx, Ry, Rz
            elif len(qids) == 1:
                gate_dict[gates[0]]["qid"] | circuit(qids[0])

            # Only coeff
            else:
                GPhase | circuit

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
        params = np.asarray(self.params)
        gamma = params[0]
        beta = params[1]
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
