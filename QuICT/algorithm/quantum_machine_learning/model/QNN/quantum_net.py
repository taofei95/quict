from typing import List, Union

from numpy_ml.neural_nets.optimizers import *

from QuICT.algorithm.quantum_machine_learning.ansatz_library import *
from QuICT.algorithm.quantum_machine_learning.encoding import *
from QuICT.algorithm.quantum_machine_learning.utils import Hamiltonian
from QuICT.algorithm.quantum_machine_learning.utils.loss import *
from QuICT.algorithm.quantum_machine_learning.utils.ml_utils import *
from QuICT.core import Circuit
from QuICT.core.gate import *

from ..model import Model


class QuantumNet(Model):
    """The quantum neural network (QNN)."""

    def __init__(
        self,
        n_qubits: int,
        ansatz: Ansatz,
        optimizer: OptimizerBase,
        data_qubits: list = None,
        hamiltonian: Union[Hamiltonian, List] = None,
        params: np.ndarray = None,
        device="GPU",
        gpu_device_id: int = 0,
        differentiator: str = "adjoint",
    ):
        """Initialize a QNN model.

        Args:
            n_qubits (int): The number of qubits.
            ansatz (Ansatz): The QNN ansatz used by the model.
            optimizer (OptimizerBase): The optimizer used to optimize the network.
            data_qubits (list, optional): List of qubits used by encoding. Defaults to None.
            hamiltonian (Union[Hamiltonian, List], optional): The hamiltonians for measurement. Defaults to None.
            params (np.ndarray, optional): Initialization parameters. Defaults to None.
            device (str, optional): The device type, one of [CPU, GPU]. Defaults to "GPU".
            gpu_device_id (int, optional): The GPU device ID. Defaults to 0.
            differentiator (str, optional): The differentiator type, one of ["adjoint", "parameter_shift].
                Defaults to "adjoint".
        """
        super(QuantumNet, self).__init__(
            n_qubits,
            optimizer,
            hamiltonian,
            params,
            device,
            gpu_device_id,
            differentiator,
        )
        self._ansatz = ansatz
        self._data_qubits = (
            list(range(n_qubits - 1)) if data_qubits is None else data_qubits
        )
        self._readout = ansatz.readout
        self._model_circuit = ansatz.init_circuit(params=params)
        self._params = ansatz.params
        self._hamiltonian = (
            [Hamiltonian([[1.0, "Z" + str(r)]]) for r in self._readout]
            if hamiltonian is None
            else hamiltonian
        )
        if isinstance(self._hamiltonian, Hamiltonian):
            self._hamiltonian = [self._hamiltonian]
        for h in self._hamiltonian:
            if not isinstance(h, Hamiltonian):
                raise ValueError

    def forward(self, data_circuits, train: bool = True):
        """The forward propagation procedure for one step.

        Args:
            data_circuits (list): Data circuits after encoding.
            train (bool, optional): Whether it is a training step, that is,
                whether to calculate the gradients and update the parameters. Defaults to True.

        Returns:
            Variable: The expectations.
        """
        state_list = []
        # FP
        for data_circuit in data_circuits:
            circuit = Circuit(self._n_qubits)
            data_circuit | circuit(self._data_qubits)
            self._model_circuit | circuit(list(range(self._n_qubits)))
            state = self._simulator.run(circuit)
            state_list.append(state)
        if train:
            # BP get expectations and d(exp) / d(params)
            # expectations = p(0>) - p(|1>)
            self._params_grads, expectations = self._differentiator.run_batch(
                circuit, self._params.copy(), state_list, self._hamiltonian
            )
        else:
            expectations = self._differentiator.get_expectations_batch(
                state_list, self._hamiltonian
            )
        return Variable(expectations)

    def backward(self, loss: Union[Variable, Loss]):
        """The backward propagation procedure for one step.

        Args:
            loss (Union[Variable, Loss]): The loss for this iteration.
        """
        self._params.zero_grad()
        for params_grads, grads in zip(self._params_grads, loss.grads):
            for params_grad, grad in zip(params_grads, grads):
                self._params.grads += grad * params_grad
        self._params.pargs = self._optimizer.update(
            self._params.pargs, self._params.grads, "params"
        )

    def update(self):
        """Update the trainable parameters in the PQC."""
        self._params.zero_grad()
        self._model_circuit.update(self._params)
