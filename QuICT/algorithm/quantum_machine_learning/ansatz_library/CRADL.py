import numpy as np

from .ansatz import Ansatz
from QuICT.core import Circuit
from QuICT.core.gate import *


class CRADL(Ansatz):
    """Color-Readout-Alternating-Double-Layer architecture (CRADL) Ansatz.
    
    Reference:
        https://arxiv.org/abs/2110.05476
    """

    def __init__(self, n_qubits: int, color_qubit: int, readout: int, layers: int):
        """Initialize a CRADL ansatz instance.

        Args:
            n_qubits (int): The number of qubits.
            color_qubit (int): The index of the color qubit.
            readout (int): The index of the readout qubit.
            layers (int, optional): The number of layers.
        """

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

    def init_circuit(self, params: Union[Variable, np.ndarray] = None):
        """Initialize a CRADL ansatz with trainable parameters.

        Args:
            params (Union[Variable, np.ndarray], optional): Initialization parameters. Defaults to None.

        Returns:
            Circuit: The CRADL ansatz.
        """

        n_data_qubits = len(self._data_qubits)
        params = (
            np.random.randn(self._layers, n_data_qubits * 2)
            if params is None
            else params
        )
        params = Variable(pargs=params) if isinstance(params, np.ndarray) else params
        if params.shape == (self._layers, n_data_qubits * 2):
            self._params = params
        else:
            raise ValueError

        circuit = Circuit(self._n_qubits)
        for l in range(self._layers):
            for i in range(n_data_qubits):
                Rxx(params[l][i]) | circuit([self._data_qubits[i], self._readout])
                Rxx(params[l][i]) | circuit([self._data_qubits[i], self._color_qubit])
            for i in range(n_data_qubits):
                Rzz(params[l][n_data_qubits + i]) | circuit(
                    [self._data_qubits[i], self._readout]
                )
                Rzz(params[l][n_data_qubits + i]) | circuit(
                    [self._data_qubits[i], self._color_qubit]
                )

        return circuit
