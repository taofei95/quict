import numpy as np

from ..model import Model
from QuICT.algorithm.quantum_machine_learning.ansatz_library import Thouless
from QuICT.algorithm.quantum_machine_learning.utils import Hamiltonian
from QuICT.algorithm.quantum_machine_learning.utils.loss import *
from QuICT.algorithm.quantum_machine_learning.utils.ml_utils import *


class HartreeFockVQENet(Model):
    """The network used by restricted Hartree-Fock VQE with Thouless ansatz
    
    Reference:
        https://arxiv.org/abs/2004.04174
    """
    def __init__(
        self,
        orbitals: int,
        electrons: int,
        hamiltonian: Hamiltonian,
        angles: np.ndarray = None,
        device="GPU",
        gpu_device_id: int = 0,
        differentiator: str = "adjoint",
    ):
        """Initialize a HartreeFockVQENet instance.

        Args:
            orbitals (int): The number of orbitals.
            electrons (int): The number of electrons.
            hamiltonian (Hamiltonian): The hamiltonian for a specific task.
        """
        super(HartreeFockVQENet, self).__init__(
            orbitals, hamiltonian, angles, device, gpu_device_id, differentiator
        )
        self.orbitals = orbitals
        self.electrons = electrons
        # Thouless ansatz
        self.ansatz = Thouless(orbitals, electrons)
        self._circuit = self.ansatz.init_circuit(angles)
        self._params = self.ansatz.params

    def run_step(self, optimizer):
        # FP
        state = self._simulator.run(self._circuit)
        # BP
        _, loss = self._differentiator.run(
            self._circuit, self._params, state, self._hamiltonian
        )
        # optimize
        self._params.pargs = optimizer.update(
            self._params.pargs, self._params.grads, "params"
        )
        self._params.zero_grad()

        # update
        self.update()
        return state, loss

    def update(self):
        self._circuit = self.ansatz.init_circuit(self._params)

    def sample(self, shots):
        sample = self._simulator.sample(shots)
        return sample
