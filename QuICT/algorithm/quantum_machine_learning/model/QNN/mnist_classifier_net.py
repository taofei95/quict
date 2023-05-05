from QuICT.core import Circuit
from QuICT.core.gate import *

from QuICT.algorithm.quantum_machine_learning.ansatz_library import *
from QuICT.algorithm.quantum_machine_learning.differentiator import Differentiator
from QuICT.algorithm.quantum_machine_learning.encoding import *
from QuICT.algorithm.quantum_machine_learning.utils import Hamiltonian
from QuICT.algorithm.quantum_machine_learning.utils.ml_utils import *
from QuICT.simulation.state_vector import StateVectorSimulator


class QuantumNet:
    def __init__(
        self,
        n_qubits: int,
        readout: int,
        layers: list = ["XX", "ZZ"],
        device="GPU",
        gpu_device_id: int = 0,
        differentiator: str = "adjoint",
    ):
        self._n_qubits = n_qubits
        self._readout = readout
        self._data_qubits = list(range(n_qubits)).remove(readout)
        self._layers = layers
        self._qnn_builder = QNNLayer(n_qubits, readout, layers)
        self._model_circuit = self._qnn_builder.init_circuit()
        self._params = self._qnn_builder.params
        self._hamiltonian = Hamiltonian([["1.0", "Z" + str(self._readout)]])

        self._simulator = StateVectorSimulator(
            device=device, gpu_device_id=gpu_device_id
        )
        self._differentiator = Differentiator(
            device=device, backend=differentiator, gpu_device_id=gpu_device_id
        )

    def hinge_loss(self, y_true, y_pred):
        y_true = 2 * y_true.type(np.float64) - 1.0
        y_pred = 2 * y_pred - 1.0
        loss = 1 - y_pred * y_true
        grads = 2.0 - 4 * y_true
        correct = np.where(y_true * y_pred > 0)[0].shape[0]
        return np.mean(loss), grads, correct

    def run_step(self, data_circuits, y_true, optimizer):
        circuit_list = []
        state_list = []
        for data_circuit in data_circuits:
            circuit = Circuit(self._n_qubits)
            data_circuit | circuit(self._data_qubits)
            self._model_circuit | circuit
            state = self._simulator.run(circuit)
            circuit_list.append(circuit)
            state_list.append(state)
        params_list, y_pred = self._differentiator.run_batch(
            circuit_list, self._params.copy(), state_list, self._hamiltonian
        )

        loss, grads, correct = self.hinge_loss(y_true, y_pred)
        for param, grad in zip(params_list, grads):
            self._params.grads += grad * param.grads
        self.params.grads /= len(circuit_list)

        # optimize
        self._params.pargs = optimizer.update(
            self._params.pargs, self._params.grads, "params"
        )
        self._params.zero_grad()
        # update
        self._model_circuit.update(self._params)

