from typing import Union

import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import H, Rxx, Ryy, Rzx, Rzz, Variable, X
from QuICT.tools.exception.algorithm import *

from .ansatz import Ansatz


class BasicQNN(Ansatz):
    """Basic QNN ansatz.

    Each layer takes the same 2-qubit gate, wich each data qubit acting on the readout qubit.
    All parametric gates have different parameters.

    For more detail, please refer to:

    References:
        `Classification with Quantum Neural Networks on Near Term Processors`
        <https://arxiv.org/abs/1802.06002>

    Note:
        By default, the last qubit is the readout qubit, and others are data qubits.

    Args:
        n_qubits (int): The number of qubits.
        layers (list): The list of PQC layers. Supported layers are "XX", "YY", "ZZ", "ZX".

    Examples:
        >>> from QuICT.algorithm.quantum_machine_learning.ansatz_library import BasicQNN
        >>> basic_qnn = BasicQNN(3, ["ZZ", "ZX"])
        >>> basic_qnn_cir = basic_qnn.init_circuit()
        >>> basic_qnn_cir.draw("command")
                                                      ┌─────────────┐
        q_0: |0>───────────■──────────────────────────┤0            ├─────────────────────
                           │                          │             │┌──────────────┐
        q_1: |0>───────────┼─────────────■────────────┤  rzx(1.639) ├┤0             ├─────
                ┌───┐┌───┐ │ZZ(0.21842)  │ZZ(0.67852) │             ││  rzx(0.7458) │┌───┐
        q_2: |0>┤ x ├┤ h ├─■─────────────■────────────┤1            ├┤1             ├┤ h ├
                └───┘└───┘                            └─────────────┘└──────────────┘└───┘
    """

    @property
    def readout(self) -> list:
        """Get the readout qubits.

        Returns:
            list: The list of readout qubits.
        """
        return [self._readout]

    def __init__(self, n_qubits: int, layers: list):
        """Initialize a basic QNN ansatz object."""
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

        Raises:
            AnsatzShapeError: An error occurred defining trainable parameters.
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
