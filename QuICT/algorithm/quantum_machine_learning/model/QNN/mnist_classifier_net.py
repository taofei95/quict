import torch
import torch.nn as nn

from QuICT.core import Circuit
from QuICT.core.gate import *
<<<<<<< HEAD
from QuICT.tools.exception.algorithm import *
from QuICT.simulation.state_vector import ConstantStateVectorSimulator
=======
from QuICT.simulation.state_vector import StateVectorSimulator
>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd

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
<<<<<<< HEAD
        """Initialize a QuantumNet instance.

        Args:
            data_qubits (int): The index of the readout qubit.
            layers (list, optional): The list of types of QNN layers.
                Currently only supports XX, YY, ZZ, and ZX. Defaults to ["XX", "ZZ"].
            encoding (str, optional): The encoding method to encode the image as quantum ansatz.
                Only support qubit encoding and amplitude encoding. Defaults to "qubit".
            device (torch.device, optional): The device to which the model is assigned.
                Defaults to torch.device("cuda:0").
        """
        super(QuantumNet, self).__init__()
        if encoding not in ["qubit", "amplitude", "FRQI","NEQR"]:
            raise QNNModelError("The encoding method should be 'qubit' or 'amplitude'")
=======
        self._n_qubits = n_qubits
        self._readout = readout
        self._data_qubits = list(range(n_qubits)).remove(readout)
>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd
        self._layers = layers

        if encoding == "qubit":
            self._encoding = Qubit(data_qubits, device)
        elif encoding == "amplitude":
<<<<<<< HEAD
            self._encoding = Amplitude(data_qubits, device)
        elif encoding == "FRQI":
            self._encoding = FRQI(device)
        elif encoding == "NEQR":
            self._encoding = NEQR(device)
        self._n_qubits = self._data_qubits + 1
        self._simulator = GpuSimulator()
        self._pqc = QNNLayer(
            list(range(self._data_qubits)), self._data_qubits, device=self._device
=======
            raise ValueError
        elif encoding == "FRQI":
            raise ValueError
        else:
            raise ValueError

        self._simulator = StateVectorSimulator(
            device=device, gpu_device_id=gpu_device_id
>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd
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
<<<<<<< HEAD
            cir_simulator = ConstantStateVectorSimulator()
            img_state = cir_simulator.run(data_circuit)
            ansatz = model_ansatz
            
=======
            cir_simulator = StateVectorSimulator()
            img_state = cir_simulator.run(data_circuit)
            ansatz = model_ansatz

>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd
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
