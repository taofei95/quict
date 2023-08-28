from typing import Union

import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import Rxx, Rzz, Variable
from QuICT.tools.exception.algorithm import *

from .ansatz import Ansatz


class CRAML(Ansatz):
    """Color-Readout-Alternating-Mixed-Layer architecture (CRAML) Ansatz.

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
        super(CRAML, self).__init__(n_qubits)
        self._color_qubit = n_qubits - 2
        self._readout = n_qubits - 1
        self._layers = layers

    def __str__(self):
        return "CRAML(n_qubits={}, layers={})".format(self._n_qubits, self._layers)

    def init_circuit(self, params: Union[Variable, np.ndarray] = None):
        """Initialize a CRAML ansatz with trainable parameters.

        Args:
            params (Union[Variable, np.ndarray], optional): Initialization parameters. Defaults to None.

        Returns:
            Circuit: The CRAML ansatz.
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
            raise AnsatzShapeError(
                str(self._layers, n_pos_qubits * 2), str(params.shape)
            )

        circuit = Circuit(self._n_qubits)
        for l in range(self._layers):
            for i, k in zip(range(n_pos_qubits), range(0, n_pos_qubits * 2, 2)):
                Rxx(params[l][k]) | circuit([i, self._readout])
                Rxx(params[l][k]) | circuit([i, self._color_qubit])
                Rzz(params[l][k + 1]) | circuit([i, self._readout])
                Rzz(params[l][k + 1]) | circuit([i, self._color_qubit])

        return circuit
