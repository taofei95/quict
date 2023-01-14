import torch

from QuICT.algorithm.quantum_machine_learning.ansatz_library import Thouless
from QuICT.algorithm.quantum_machine_learning.model.VQA import VQENet
from QuICT.algorithm.quantum_machine_learning.utils import Hamiltonian


class HartreeFockVQENet(VQENet):
    """The network used by Hartree-Fock VQE with Thouless ansatz
    
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
        self.orbitals = orbitals
        self.electrons = electrons
        self.ansatz = Thouless(device)
        super().__init__(orbitals, electrons, hamiltonian, device)

    def define_network(self):
        self.params = torch.nn.Parameter(
            torch.zeros(self.electrons * (self.orbitals - self.electrons), device=self.device),
            requires_grad=True
        )

    def forward(self, state=None):
        """The forward propagation process of QAOANet.

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
