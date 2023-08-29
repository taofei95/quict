from typing import Union

import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import H, Rxx, Ryy, Rzx, Rzz, Variable, X
from QuICT.tools.exception.algorithm import *

from .ansatz import Ansatz


class BasicQNN(Ansatz):
    """Basic QNN ansatz.

    References:
        https://arxiv.org/abs/1802.06002
    """

    @property
    def readout(self):
        return [self._readout]

    def __init__(self, n_qubits: int, layers: list):
        """Initialize a basic QNN instance.

        Args:
            n_qubits (int): The number of qubits.
            layers (list): The list of PQC layers. Supported layers are "XX", "YY", "ZZ", "ZX".
        """
        super(BasicQNN, self).__init__(n_qubits)
        self._readout = n_qubits - 1
        self._data_qubits = list(range(n_qubits - 1))
        self._layers = layers
        self._validate_layers()

    def __str__(self):
        return "BasicQNN(n_qubits={}, layers={})".format(self._n_qubits, self._layers)

    def init_circuit(self, params: Union[Variable, np.ndarray] = None):
        """Initialize a basic QNN ansatz with trainable parameters.

        Args:
            params (Union[Variable, np.ndarray], optional): Initialization parameters. Defaults to None.

        Returns:
            Circuit: The basic QNN ansatz.
        """
        params = (
            np.random.randn(len(self._layers), self._n_qubits - 1)
            if params is None
            else params
        )
        params = Variable(pargs=params) if isinstance(params, np.ndarray) else params
        if params.shape == (len(self._layers), self._n_qubits - 1):
            self._params = params
        else:
            raise AnsatzShapeError(
                str((len(self._layers), self._n_qubits - 1)), str(params.shape)
            )

        gate_dict = {"XX": Rxx, "YY": Ryy, "ZZ": Rzz, "ZX": Rzx}
        circuit = Circuit(self._n_qubits)
        X | circuit(self._readout)
        H | circuit(self._readout)
        for l, gate in zip(range(len(self._layers)), self._layers):
            if gate not in gate_dict.keys():
                raise AnsatzValueError('["XX", "YY", "ZZ", "ZX"]', gate)

            for i in range(self._n_qubits - 1):
                gate_dict[gate](params[l][i]) | circuit(
                    [self._data_qubits[i], self._readout]
                )
        H | circuit(self._readout)

        return circuit

    def _validate_layers(self):
        for layer in self._layers:
            if layer not in ["XX", "YY", "ZZ", "ZX"]:
                raise AnsatzValueError('["XX", "YY", "ZZ", "ZX"]', layer)
