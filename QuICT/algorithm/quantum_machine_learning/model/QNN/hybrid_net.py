from ..model import Model
from QuICT.core import Circuit
from QuICT.core.gate import *


from QuICT.algorithm.quantum_machine_learning.ansatz_library import *
from QuICT.algorithm.quantum_machine_learning.encoding import *
from QuICT.algorithm.quantum_machine_learning.utils import Hamiltonian
from QuICT.algorithm.quantum_machine_learning.utils.loss import *
from QuICT.algorithm.quantum_machine_learning.utils.ml_utils import *


class HybridNet(Model):
    def __init__(
        self,
        n_qubits: int,
        encoding,
        qnn_builder: Ansatz,
        hamiltonian: Hamiltonian = None,
        params: np.ndarray = None,
        device="GPU",
        gpu_device_id: int = 0,
        differentiator: str = "adjoint",
    ):
        super(HybridNet, self).__init__(
            n_qubits, hamiltonian, params, device, gpu_device_id, differentiator
        )
        self._encoding = encoding
        self._data_qubits = encoding.data_qubits
        self._qnn_builder = qnn_builder
        self._model_circuit = self._qnn_builder.init_circuit(params=params)
        self._params = self._qnn_builder.params
        if hamiltonian is None:
            hamiltonian_list = ["Z" + str(i) for i in range(n_qubits)]
            self._hamiltonian = Hamiltonian([[1.0] + hamiltonian_list])
        else:
            self._hamiltonian = hamiltonian

    def run_step(
        self, data_batch, y_true, optimizer, loss_fun: Loss, train: bool = True
    ):
        data_circuits = [self._encoding(x) for x in data_batch]
        circuit_list = []
        state_list = []
        # Quantum part
        # FP
        for data_circuit in data_circuits:
            circuit = Circuit(self._n_qubits)
            data_circuit | circuit(self._data_qubits)
            self._model_circuit | circuit(list(range(self._n_qubits)))
            state = self._simulator.run(circuit)
            circuit_list.append(self._model_circuit)
            state_list.append(state)
        if train:
            # BP get expectations and d(exp) / d(params)
            params_grads, poss = self._differentiator.run_batch(
                circuit_list, self._params.copy(), state_list, self._hamiltonian
            )
        else:
            poss = self._differentiator.get_expectations_batch(
                state_list, self._hamiltonian
            )

        y_true = 2 * y_true - 1.0
        y_pred = -poss
        loss = loss_fun(y_pred, y_true)
        correct = np.where(y_true * y_pred > 0)[0].shape[0]

        if train:
            # BP get loss and d(loss) / d(exp)
            grads = -loss_fun.gradient()
            # BP get d(loss) / d(params)
            for params_grad, grad in zip(params_grads, grads):
                self._params.grads += grad * params_grad

            # optimize
            self._params.pargs = optimizer.update(
                self._params.pargs, self._params.grads, "params"
            )
            self._params.zero_grad()
            # update
            self.update()

        return loss, correct

    def update(self):
        self._model_circuit.update(self._params)

