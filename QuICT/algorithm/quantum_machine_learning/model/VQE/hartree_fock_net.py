import numpy as np
from numpy_ml.neural_nets.optimizers import *

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
        optimizer: OptimizerBase,
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
            orbitals,
            optimizer,
            hamiltonian,
            angles,
            device,
            gpu_device_id,
            differentiator
        )
        self.orbitals = orbitals
        self.electrons = electrons
        # Thouless ansatz
        self.ansatz = Thouless(orbitals, electrons)
        self._circuit = self.ansatz.init_circuit(angles)
        self._params = self.ansatz.params

    def run(self):
        """Train HFVQE for one step.

        Returns:
            np.float: The loss for this iteration.
        """
        loss = self.forward(train=True)
        self.backward(loss)
        self.update()
        return loss.item

    def forward(self, train=True):
        """The forward propagation procedure for one step.

        Args:
            train (bool, optional): Whether it is a training step, that is,
                whether to calculate the gradients and update the parameters. Defaults to True.

        Returns:
            Variable: The expectation.
        """
        # FP
        state = self._simulator.run(self._circuit)
        if train:
            # BP
            self._params_grads, expectation = self._differentiator.run(
                self._circuit, self._params, state, [self._hamiltonian]
            )
        else:
            expectation = self._differentiator.get_expectations(
                self._circuit, state, [self._hamiltonian]
            )
        return Variable(expectation)

    def backward(self, loss: Union[Variable, Loss]):
        """The backward propagation procedure for one step.

        Args:
            loss (Union[Variable, Loss]): The loss for this iteration.
        """
        self._params.zero_grad()
        for params_grad, grad in zip(self._params_grads, loss.grads):
            self._params.grads += grad * params_grad
        self._params.pargs = self._optimizer.update(
            self._params.pargs, self._params.grads, "params"
        )

    def update(self):
        """Update the trainable parameters in the PQC."""
        self._params.zero_grad()
        self._circuit = self.ansatz.init_circuit(self._params)

    def sample(self, shots: int):
        """Sample the measured result from current state vector.

        Args:
            shots (int): The sample times for current state vector.

        Returns:
             List[int]: The measured result list.
        """
        sample = self._simulator.sample(shots)
        return sample
