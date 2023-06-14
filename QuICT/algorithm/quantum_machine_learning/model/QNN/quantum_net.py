from ..model import Model
from QuICT.core import Circuit
from QuICT.core.gate import *

import time

from QuICT.algorithm.quantum_machine_learning.ansatz_library import *
from QuICT.algorithm.quantum_machine_learning.encoding import *
from QuICT.algorithm.quantum_machine_learning.utils import Hamiltonian
from QuICT.algorithm.quantum_machine_learning.utils.loss import *
from QuICT.algorithm.quantum_machine_learning.utils.ml_utils import *


class QuantumNet(Model):
    def __init__(
        self,
        n_qubits: int,
        readout: int,
        layers: list = ["XX", "ZZ"],
        hamiltonian: Hamiltonian = None,
        params: np.ndarray = None,
        device="GPU",
        gpu_device_id: int = 0,
        differentiator: str = "adjoint",
    ):
        super(QuantumNet, self).__init__(
            n_qubits, hamiltonian, params, device, gpu_device_id, differentiator
        )
        self._readout = readout
        self._data_qubits = list(range(n_qubits))
        self._data_qubits.remove(readout)
        self._qnn_builder = QNNLayer(n_qubits, readout, layers)
        # self._qnn_builder = CRADL(n_qubits, n_qubits - 2, readout, layers)
        self._model_circuit = self._qnn_builder.init_circuit(params=params)
        self._params = self._qnn_builder.params
        self._hamiltonian = (
            Hamiltonian([[1.0, "Z" + str(self._readout)]])
            if hamiltonian is None
            else hamiltonian
        )

    def run_step(
        self, data_circuits, y_true, optimizer, loss_fun: Loss, train: bool = True
    ):
        circuit_list = []
        state_list = []
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

