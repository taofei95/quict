import numpy as np

from .ansatz import Ansatz
from QuICT.core import Circuit
from QuICT.core.gate import *


class CRADL(Ansatz):
    """IMAGE COMPRESSION AND CLASSIFICATION USING QUBITS AND QUANTUM DEEP LEARNING
    """

    def __init__(self, n_qubits: int, color_qubit: int, readout: int, layers: list):
        super(CRADL, self).__init__(n_qubits)
        if (
            readout < 0
            or readout >= self._n_qubits
            or color_qubit < 0
            or color_qubit >= self._n_qubits
            or readout == color_qubit
        ):
            raise ValueError
        self._data_qubits = list(range(n_qubits))
        self._data_qubits.remove(readout)
        self._data_qubits.remove(color_qubit)
        self._color_qubit = color_qubit
        self._readout = readout
        self._layers = layers
        self._validate_layers()

    def init_circuit(self, params: Union[Variable, np.ndarray] = None):
        params = (
            np.random.randn(len(self._layers), len(self._data_qubits) * 2)
            if params is None
            else params
        )
        params = Variable(pargs=params) if isinstance(params, np.ndarray) else params
        if params.shape == (len(self._layers), len(self._data_qubits) * 2):
            self._params = params
        else:
            raise ValueError

        gate_dict = {"XX": Rxx, "YY": Ryy, "ZZ": Rzz, "ZX": Rzx}
        circuit = Circuit(self._n_qubits)
        for l, gate in zip(range(len(self._layers)), self._layers):
            if gate not in gate_dict.keys():
                raise ValueError

            for i in range(self._n_qubits - 2):
                gate_dict[gate](params[l][i]) | circuit(
                    [self._data_qubits[i], self._readout]
                )
                gate_dict[gate](params[l][self._n_qubits - 2 + i]) | circuit(
                    [self._data_qubits[i], self._color_qubit]
                )

        return circuit

    def _validate_layers(self):
        for layer in self._layers:
            if layer not in ["XX", "YY", "ZZ", "ZX"]:
                raise ValueError


if __name__ == "__main__":
    cradl = CRADL(6, 4, 5, ["ZZ"])
    circuit = cradl.init_circuit()
    circuit.draw(filename="q")
