from numpy_ml.neural_nets.optimizers import *

from ..model import Model
from QuICT.core import Circuit
from QuICT.core.gate import *

from QuICT.algorithm.quantum_machine_learning.ansatz_library import *
from QuICT.algorithm.quantum_machine_learning.encoding import *
from QuICT.algorithm.quantum_machine_learning.utils import Hamiltonian
from QuICT.algorithm.quantum_machine_learning.utils.loss import *
from QuICT.algorithm.quantum_machine_learning.utils.ml_utils import *


class QuantumNet(Model):
    """The quantum neural network (QNN)."""

    def __init__(
        self,
        n_qubits: int,
        ansatz: Ansatz,
        optimizer: OptimizerBase,
        loss_fun: Loss,
        data_qubits: list = None,
        hamiltonian: Hamiltonian = None,
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
            loss_fun (Loss): The loss function used by the model.
            data_qubits (list, optional): List of qubits used by encoding. Defaults to None.
            hamiltonian (Hamiltonian, optional): The hamiltonian for measurement. Defaults to None.
            params (np.ndarray, optional): Initialization parameters. Defaults to None.
            device (str, optional): The device type, one of [CPU, GPU]. Defaults to "GPU".
            gpu_device_id (int, optional): The GPU device ID. Defaults to 0.
            differentiator (str, optional): The differentiator type, one of ["adjoint", "parameter_shift]. Defaults to "adjoint".
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
        self._loss_fun = loss_fun
        self._data_qubits = (
            list(range(n_qubits - 1)) if data_qubits is None else data_qubits
        )
        self._readout = ansatz.readout
        self._model_circuit = ansatz.init_circuit(params=params)
        self._params = ansatz.params
        self._hamiltonian = (
            Hamiltonian([[1.0, "Z" + str(r)] for r in self._readout])
            if hamiltonian is None
            else hamiltonian
        )

    def run_step(self, data_circuits, y_true, train: bool = True):
        """Train QNN for one step.

        Args:
            data_circuits (list): Data circuits after encoding.
            y_true (np.ndarry): The ground truth.
            train (bool, optional): Whether it is a training step, that is, whether to calculate the gradients and update the parameters. Defaults to True.

        Returns:
            np.float: The loss.
            int: The number of correctly classified instances.
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
            params_grads, poss = self._differentiator.run_batch(
                circuit, self._params.copy(), state_list, self._hamiltonian
            )
        else:
            poss = self._differentiator.get_expectations_batch(
                state_list, self._hamiltonian
            )

        y_true = 2 * y_true - 1.0
        y_pred = -poss
        loss = self._loss_fun(y_pred, y_true)
        correct = np.where(y_true * y_pred > 0)[0].shape[0]

        if train:
            # BP get loss and d(loss) / d(exp)
            grads = -self._loss_fun.gradient()
            # BP get d(loss) / d(params)
            for params_grad, grad in zip(params_grads, grads):
                self._params.grads += grad * params_grad

            # optimize
            self._params.pargs = self._optimizer.update(
                self._params.pargs, self._params.grads, "params"
            )
            self._params.zero_grad()
            # update
            self._update()

        return loss, correct

    def _update(self):
        self._model_circuit.update(self._params)
