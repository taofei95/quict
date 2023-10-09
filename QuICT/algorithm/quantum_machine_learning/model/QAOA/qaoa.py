from typing import Union

from numpy_ml.neural_nets.optimizers import *

from QuICT.algorithm.quantum_machine_learning.ansatz_library import *
from QuICT.algorithm.quantum_machine_learning.utils import Hamiltonian, Loss
from QuICT.algorithm.quantum_machine_learning.utils.ml_utils import *

from ..model import Model


class QAOA(Model):
    """The quantum approximate optimization algorithm (QAOA)."""

    def __init__(
        self,
        n_qubits: int,
        p: int,
        hamiltonian: Hamiltonian,
        optimizer: OptimizerBase,
        params: np.ndarray = None,
        device: str = "CPU",
        gpu_device_id: int = 0,
        differentiator: str = "adjoint",
    ):
        """Initialize a QAOA model.

        Args:
            n_qubits (int): The number of qubits.
            p (int): The number of layers of the QAOA network.
            hamiltonian (Hamiltonian): The hamiltonian for a specific problem.
            optimizer (OptimizerBase): The optimizer used to optimize the network.
            params (np.ndarray, optional): Initialization parameters. Defaults to None.
            device (str, optional): The device type, one of [CPU, GPU]. Defaults to "CPU".
            gpu_device_id (int, optional): The GPU device ID. Defaults to 0.
            differentiator (str, optional): The differentiator type, one of ["adjoint", "parameter_shift].
                Defaults to "adjoint".
        """
        super(QAOA, self).__init__(
            n_qubits,
            optimizer,
            hamiltonian,
            params,
            device,
            gpu_device_id,
            differentiator,
        )
        self._qaoa_builder = QAOALayer(n_qubits, p, hamiltonian)
        self._circuit = self._qaoa_builder.init_circuit(params=params)
        self._params = self._qaoa_builder.params

    def run(self):
        """Train QAOA for one step.

        Returns:
            np.float: The loss for this iteration.
        """
        expectation = self.forward(train=True)
        loss = -expectation
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
        self._circuit = self._qaoa_builder.init_circuit(self._params)

    def sample(self, shots: int):
        """Sample the measured result from current state vector.

        Args:
            shots (int): The sample times for current state vector.

        Returns:
             List[int]: The measured result list.
        """
        sample = self._simulator.sample(shots)
        return sample
