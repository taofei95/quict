import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import *


class QNNLayer:
    """Initialize a QNNLayer instance."""

    @property
    def params(self):
        return self._params

    @property
    def circuit(self):
        return self._circuit

    def __init__(self, n_qubits: int, readout: int, layers: list):
        """The QNN layer constructor.
        """
        self._n_qubits = n_qubits
        if readout < 0 or readout >= self._n_qubits:
            raise ValueError
        self._data_qubits = list(range(n_qubits))
        self._data_qubits.remove(readout)
        self._readout = readout
        self._layers = layers
        self._params = None
        self._circuit = None

    def init_circuit(self, params: Union[Variable, np.ndarray] = None):
        params = (
            Variable(np.random.randn(len(self._layers), self._n_qubits - 1))
            if params is None
            else params
        )
        if params.shape == (len(self._layers), self._n_qubits - 1):
            self._params = params
        else:
            raise ValueError

        gate_dict = {"XX": Rxx, "YY": Ryy, "ZZ": Rzz, "ZX": Rzx}
        self._circuit = Circuit(self._n_qubits)
        X | self._circuit(self._readout)
        H | self._circuit(self._readout)
        for l, gate in zip(range(len(self._layers)), self._layers):
            if gate not in gate_dict.keys():
                raise ValueError

            for i in range(self._n_qubits - 1):
                gate_dict[gate](params[l][i]) | self._circuit(
                    [self._data_qubits[i], self._readout]
                )
        H | self._circuit(self._readout)

        return self._circuit
