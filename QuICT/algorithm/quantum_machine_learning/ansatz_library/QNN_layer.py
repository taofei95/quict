import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.tools.exception.algorithm import *


class QNNLayer:
    """Initialize a QNNLayer instance."""

<<<<<<< HEAD
    def __init__(self, data_qubits, result_qubit, device=torch.device("cuda:0")):
        """The QNN layer constructor.

        Args:
            data_qubits (list): The list of the data qubits indexes.
            result_qubit (int): The index of the readout qubit.
            device (str, optional): The device to which the model is assigned.
                Defaults to "cuda:0".
        """
        self._n_qubits = len(data_qubits) + 1
        if (
            result_qubit < 0
            or result_qubit >= self._n_qubits
            or result_qubit in data_qubits
        ):
            raise QNNModelError("Wrong result qubit.")
        self._data_qubits = data_qubits
        self._result_qubit = result_qubit
        self._device = device

    def __call__(self, two_qubit_gates, params):
        """Build specified QNN layer ansatz with trainable parameters.

        Args:
            two_qubit_gates (str or list): The types of QNN layers.
                Currently only supports XX, YY, ZZ, and ZX.
            params (torch.nn.parameter): The parameters to be trained.
=======
    __DEVICE = ["CPU", "GPU"]
>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd

    def __init__(self, n_qubits: int, readout: int):
        """The QNN layer constructor.
        """
<<<<<<< HEAD
        if not isinstance(two_qubit_gates, list):
            two_qubit_gates = [two_qubit_gates]
        n_layers = len(two_qubit_gates)
        if params.shape[0] != n_layers or params.shape[1] != self._n_qubits - 1:
            raise QNNModelError(
                "The shape of the parameters should be [n_layers, n_data_qubits]."
            )

        gate_dict = {
            "XX": Rxx_tensor,
            "YY": Ryy_tensor,
            "ZZ": Rzz_tensor,
            "ZX": Rzx_tensor,
        }
        ansatz = Ansatz(self._n_qubits, device=self._device)
        for l, gate in zip(range(n_layers), two_qubit_gates):
            if gate not in gate_dict.keys():
                raise QNNModelError(
                    "Invalid Two Qubit Gate. Should be XX, YY, ZZ or ZX."
                )

            for i in range(self._n_qubits - 1):
                ansatz.add_gate(
                    gate_dict[gate](params[l][i]),
                    [self._data_qubits[i], self._result_qubit],
                )
        return ansatz

    def circuit_layer(self, two_qubit_gates, params):
        """Build specified QNN layer circuit.

        Args:
            two_qubit_gates (str or list): The types of QNN layers.
                Currently only supports XX, YY, ZZ, and ZX.
            params (torch.nn.parameter): The parameters.

        Returns:
            Circuit: The QNNLayer circuit.
        """
        if not isinstance(two_qubit_gates, list):
            two_qubit_gates = [two_qubit_gates]
        n_layers = len(two_qubit_gates)
        params = params.cpu().detach().numpy().astype(np.float64)
        if params.shape[0] != n_layers or params.shape[1] != self._n_qubits - 1:
            raise QNNModelError(
                "The shape of the parameters should be [n_layers, n_data_qubits]."
            )
=======
        self._n_qubits = n_qubits
        if readout < 0 or readout >= self._n_qubits:
            raise ValueError
        self._data_qubits = list(range(n_qubits)).remove(readout)
        self._readout = readout

    def __call__(self, two_qubit_gates, params: Variable):
        if not isinstance(two_qubit_gates, list):
            two_qubit_gates = [two_qubit_gates]
        n_layers = len(two_qubit_gates)
        if params.shape[0] != n_layers or params.shape[1] != self._n_qubits - 1:
            raise ValueError
>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd

        gate_dict = {"XX": Rxx, "YY": Ryy, "ZZ": Rzz, "ZX": Rzx}
        circuit = Circuit(self._n_qubits)
        for l, gate in zip(range(n_layers), two_qubit_gates):
            if gate not in gate_dict.keys():
<<<<<<< HEAD
                raise QNNModelError(
                    "Invalid Two Qubit Gate. Should be XX, YY, ZZ or ZX."
                )
=======
                raise ValueError
>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd

            for i in range(self._n_qubits - 1):
                gate_dict[gate](params[l][i]) | circuit(
                    [self._data_qubits[i], self._readout]
                )
        return circuit
