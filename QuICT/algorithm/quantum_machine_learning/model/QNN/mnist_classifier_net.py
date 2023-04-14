import torch
import torch.nn as nn

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.simulation.state_vector import StateVectorSimulator

# from QuICT.algorithm.quantum_machine_learning.ansatz_library import QNNLayer
# from QuICT.algorithm.quantum_machine_learning.utils.encoding import *


class QuantumNet:
    def __init__(
        self,
        n_qubits,
        readout,
        layers=["XX", "ZZ"],
        encoding="qubit",
        device="GPU",
        gpu_device_id: int = 0,
    ):
        self._n_qubits = n_qubits
        self._readout = readout
        self._data_qubits = list(range(n_qubits)).remove(readout)
        self._layers = layers

        if encoding == "qubit":
            self._encoding = Qubit(data_qubits, device)
        elif encoding == "amplitude":
            raise ValueError
        elif encoding == "FRQI":
            raise ValueError
        else:
            raise ValueError

        self._simulator = StateVectorSimulator(
            device=device, gpu_device_id=gpu_device_id
        )
        self._model_circuit = QNNLayer(n_qubits, readout)
        self._params = Variable(pargs=np.random.rand(len(self._layers), self._n_qubits))

    def forward(self, X):
        Y_pred = torch.zeros([X.shape[0]], device=self._device)
        for i in range(X.shape[0]):
            self._encoding.encoding(X[i])
            # data_ansatz = self._encoding.ansatz
            model_ansatz = self._construct_ansatz()
            # ansatz = data_ansatz + model_ansatz
            data_circuit = self._encoding.circuit
            cir_simulator = StateVectorSimulator()
            img_state = cir_simulator.run(data_circuit)
            ansatz = model_ansatz

            if self._device.type == "cpu":
                state = ansatz.forward()
                prob = ansatz.measure_prob(self._data_qubits, state)
            else:
                state = self._simulator.forward(ansatz, state=img_state)
                prob = self._simulator.measure_prob(self._data_qubits, state)
            if prob is None:
                raise QNNModelError("There is no Measure Gate on the readout qubit.")
            Y_pred[i] = prob[1]
        return Y_pred

    def _construct_circuit(self):
        """Build the model circuit."""
        model_circuit = Circuit(self._n_qubits)
        X | model_circuit(self._data_qubits)
        H | model_circuit(self._data_qubits)
        sub_circuit = self._pqc.circuit_layer(self._layers, self.params)
        model_circuit.extend(sub_circuit.gates)
        H | model_circuit(self._data_qubits)
        return model_circuit
