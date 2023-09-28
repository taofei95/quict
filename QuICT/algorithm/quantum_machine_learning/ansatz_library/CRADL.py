from typing import Union

import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import Rxx, Rzz, Variable
from QuICT.tools.exception.algorithm import *

from .ansatz import Ansatz


class CRADL(Ansatz):
    """Color-Readout-Alternating-Double-Layer architecture (CRADL) Ansatz.

    Consist of consecutive data-readout data-color XX gates, followed by analogous ZZ gates.

    For more detail, please refer to:

    Reference:
        `Image Compression and Classification Using Qubits and Quantum Deep Learning`
        <https://arxiv.org/abs/2110.05476>.

    Note:
        Only applicable to FRQI encoding or NEQR for binary images. By default, the last qubit is the readout qubit,
        the penultimate qubit is the color qubit, and others are data qubits.

    Args:
        n_qubits (int): The number of qubits.
        layers (int): The number of layers.

    Examples:
        >>> from QuICT.algorithm.quantum_machine_learning.ansatz_library import CRADL
        >>> cradl = CRADL(4, 1)
        >>> cradl_cir = cradl.init_circuit()
        >>> cradl_cir.draw("command")
                ┌───────────────┐┌───────────────┐
        q_0: |0>┤0              ├┤0              ├─────────────────────────────────────■─────────────■────────────────────────────────────────────
                │               ││               │┌────────────────┐┌────────────────┐ │             │
        q_1: |0>┤               ├┤  rxx(-0.6626) ├┤0               ├┤0               ├─┼─────────────┼─────────────■───────────────■──────────────
                │  rxx(-0.6626) ││               ││                ││  rxx(-0.27888) │ │             │ZZ(-1.6315)  │               │ZZ(0.0095895)
        q_2: |0>┤               ├┤1              ├┤  rxx(-0.27888) ├┤1               ├─┼─────────────■─────────────┼───────────────■──────────────
                │               │└───────────────┘│                │└────────────────┘ │ZZ(-1.6315)                │ZZ(0.0095895)
        q_3: |0>┤1              ├─────────────────┤1               ├───────────────────■───────────────────────────■──────────────────────────────
                └───────────────┘                 └────────────────┘
    """

    @property
    def readout(self) -> list:
        """Get the readout qubits.

        Returns:
            list: The list of readout qubits.
        """
        return [self._readout]

    def __init__(self, n_qubits: int, layers: int):
        """Initialize a CRADL ansatz instance."""
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

        Raises:
            AnsatzShapeError: An error occurred defining trainable parameters.
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
            for i in range(n_pos_qubits):
                Rxx(params[l][i]) | circuit([i, self._readout])
                Rxx(params[l][i]) | circuit([i, self._color_qubit])
            for i in range(n_pos_qubits):
                Rzz(params[l][n_pos_qubits + i]) | circuit([i, self._readout])
                Rzz(params[l][n_pos_qubits + i]) | circuit([i, self._color_qubit])

        return circuit
