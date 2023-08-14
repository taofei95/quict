from typing import Union

import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import CX, CZ, CRy, Ry, Rz, Variable
from QuICT.tools.exception.algorithm import *

from .ansatz import Ansatz


class HEAnsatz(Ansatz):
    """Hardware-Efficient Ansatz.

    References:
        https://www.nature.com/articles/nature23879
    """

    @property
    def readout(self):
        return self._readout

    def __init__(self, n_qubits: int, d: int, layers: list, readout: list = None):
        """Initialize an HE-ansatz instance.

        Args:
            n_qubits (int): The number of qubits.
            d (int): The depth of HE-ansatz.
            layers (list): The list of layers. Supported layers are "CX", "CZ", "CRy", "RY", "RZ".
            readout (list, optional): The readout qubits. Defaults to None.
        """
        super(HEAnsatz, self).__init__(n_qubits)
        self._d = d
        self._layers = layers
        self._param_layers = 0
        self._readout = [0] if readout is None else readout
        self._validate_layers()

    def __str__(self):
        return "HEAnsatz(n_qubits={}, d={}, layers={})".format(
            self._n_qubits, self._d, self._layers
        )

    def init_circuit(self, params: Union[Variable, np.ndarray] = None):
        """Initialize an HE-ansatz with trainable parameters.

        Args:
            params (Union[Variable, np.ndarray], optional): Initialization parameters. Defaults to None.

        Returns:
            Circuit: The HE-ansatz ansatz.
        """
        params = (
            np.random.randn(self._d, self._param_layers, self._n_qubits)
            if params is None
            else params
        )
        params = Variable(pargs=params) if isinstance(params, np.ndarray) else params
        if params.shape == (self._d, self._param_layers, self._n_qubits):
            self._params = params
        else:
            raise AnsatzShapeError(
                str(self._d, self._param_layers, self._n_qubits), str(params.shape)
            )

        gate_dict = {"CX": CX, "CZ": CZ, "CRy": CRy, "RY": Ry, "RZ": Rz}
        circuit = Circuit(self._n_qubits)
        for i in range(self._d):
            param_layer = 0
            for gate in self._layers:
                if gate in ["RY", "RZ"]:
                    for qid in range(self._n_qubits):
                        gate_dict[gate](params[i][param_layer][qid]) | circuit(qid)
                    param_layer += 1
                elif gate in ["CRy"]:
                    for qid in range(self._n_qubits - 1):
                        gate_dict[gate](params[i][param_layer][qid]) | circuit(
                            [qid, qid + 1]
                        )
                    gate_dict[gate](
                        params[i][param_layer][self._n_qubits - 1]
                    ) | circuit([self._n_qubits - 1, 0])
                    param_layer += 1
                else:
                    for qid in range(self._n_qubits - 1):
                        gate_dict[gate] | circuit([qid, qid + 1])
                    gate_dict[gate] | circuit([self._n_qubits - 1, 0])

        return circuit

    def _validate_layers(self):
        for layer in self._layers:
            if layer in ["RY", "RZ", "CRy"]:
                self._param_layers += 1
            elif layer not in ["CX", "CZ"]:
                raise AnsatzValueError('["RY", "RZ", "CRy", "CX", "CZ"]', layer)
