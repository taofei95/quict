import torch
import numpy as np

from QuICT.algorithm.quantum_machine_learning.ansatz_library import Thouless
from QuICT.algorithm.quantum_machine_learning.utils import GpuSimulator, Hamiltonian


class HartreeFockVQENet(torch.nn.Module):
    """The network used by restricted Hartree-Fock VQE with Thouless ansatz
    
    Reference:
        https://arxiv.org/abs/2004.04174
    """
    def __init__(
        self,
        orbitals: int,
        electrons: int,
        hamiltonian: Hamiltonian,
        device=torch.device("cuda:0"),
    ):
        """Initialize a HartreeFockVQENet instance.

        Args:
            orbitals (int): The number of orbitals.
            electrons (int): The number of electrons.
            hamiltonian (Hamiltonian): The hamiltonian for a specific task.
            device (torch.device, optional): The device to which the HartreeFockVQENet is assigned.
                Defaults to torch.device("cuda:0").
        """
        super().__init__()
        # In restricted Hartree-Fock VQE, we assert the electrons are always in pairs.
        self.orbitals = orbitals // 2
        self.electrons = electrons // 2
        # Thouless ansatz
        self.ansatz = Thouless(device)
        self.hamiltonian = hamiltonian
        self.device = device
        self.simulator = GpuSimulator()
        self.define_network()

    def define_network(self):
        self.params = torch.nn.Parameter(
            torch.zeros(self.electrons * (self.orbitals - self.electrons), device=self.device),
            requires_grad=True
        )

    def forward(self, state=None):
        """The forward propagation process of HartreeFockVQENet.

        Args:
            state (np.array/torch.Tensor, optional): The input state vector.
                Defaults to None, which means the initial state |0>.

        Returns:
            torch.Tensor: The output state vector.
        """
        ansatz = self.ansatz(self.orbitals, self.electrons, self.params)
        if self.device.type == "cpu":
            state = ansatz.forward(state)
        else:
            state = self.simulator.forward(ansatz, state)
        return state

    def loss_func(self, state):
        """The loss function for VQE, which aims to minimize the expectation of H.

        Args:
            state (torch.Tensor): The state vector.

        Returns:
            torch.Tensor: Loss, which is equal to the expectation of H.
        """
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(self.device)
        assert state.shape[0] == 1 << self.orbitals
        expanded_state = self._expand_state(state)

        ansatz_list = self.hamiltonian.construct_hamiton_ansatz(
            2 * self.orbitals, self.device
        )
        coefficients = self.hamiltonian.coefficients
        state_vector = torch.zeros(1 << (2 * self.orbitals), dtype=torch.complex128).to(
            self.device
        )
        for coeff, ansatz in zip(coefficients, ansatz_list):
            sv = ansatz.forward(expanded_state)
            state_vector += coeff * sv
        loss = torch.sum(expanded_state.conj() * state_vector).real

        return loss

    def _expand_state(self, state: torch.Tensor):
        """
        Expand the restricted state back to the original one

        Args:
            state(torch.Tensor): the restricted state vector

        Returns:
            torch.Tensor: expanded state vector
        """
        size = state.shape[0]
        expanded = torch.zeros(size * size, dtype=torch.complex128, device=self.device)
        idx = []
        for i in range(size):
            i_bin = np.binary_repr(i, width=self.orbitals)
            i_double = ''
            for n in i_bin:
                i_double += (n + n)
            idx.append(int(i_double, 2))
        expanded[idx] = state
        return expanded
