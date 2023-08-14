from typing import Union

import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import Rxx, Rzz, Variable

from .ansatz import Ansatz


class CRADL(Ansatz):
    """Color-Readout-Alternating-Double-Layer architecture (CRADL) Ansatz.

    **Only applicable to the encoding methods with only one color qubit.**

    Reference:
        https://arxiv.org/abs/2110.05476
    """

    @property
    def readout(self):
        return [self._readout]

    def __init__(self, n_qubits: int, layers: int):
        """Initialize a CRADL ansatz instance.

        Args:
            n_qubits (int): The number of qubits.
            layers (int): The number of layers.
        """
        super(CRADL, self).__init__(n_qubits)
        self._color_qubit = n_qubits - 2
        self._readout = n_qubits - 1
        self._layers = layers

    def __str__(self):
        return "CRADL(n_qubits={}, layers={})".format(self._n_qubits, self._layers)

    def init_circuit(self, params: Union[Variable, np.ndarray] = None):
        """Initialize a CRADL ansatz with trainable parameters.

        Args:
            params (Union[Variable, np.ndarray], optional): Initialization parameters. Defaults to None.

        Returns:
            Circuit: The CRADL ansatz.
        """
        n_pos_qubits = self._n_qubits - 2
        params = (
            np.random.randn(self._layers, n_pos_qubits * 2)
            if params is None
            else params
        )
        params = Variable(pargs=params) if isinstance(params, np.ndarray) else params
        if params.shape == (self._layers, n_pos_qubits * 2):
            self._params = params
        else:
            raise ValueError

        circuit = Circuit(self._n_qubits)
        for l in range(self._layers):
            for i in range(n_pos_qubits):
                Rxx(params[l][i]) | circuit([i, self._readout])
                Rxx(params[l][i]) | circuit([i, self._color_qubit])
            for i in range(n_pos_qubits):
                Rzz(params[l][n_pos_qubits + i]) | circuit([i, self._readout])
                Rzz(params[l][n_pos_qubits + i]) | circuit([i, self._color_qubit])

        return circuit
