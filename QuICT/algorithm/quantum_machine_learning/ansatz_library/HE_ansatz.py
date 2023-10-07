from typing import Union

import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import CX, CZ, CRy, Ry, Rz, Variable
from QuICT.tools.exception.algorithm import *

from .ansatz import Ansatz


class HEAnsatz(Ansatz):
    """Hardware-Efficient Ansatz.

    Consisting of alternating parameterized layers which is composed of single-qubit gates
    and entanglers which is composed of 2-qubit gates.

    For more detail, please refer to:

    References:
        `Hardware-efficient variational quantum eigensolver for small molecules and quantum magnets`
        <https://www.nature.com/articles/nature23879>

    Note:
        The currently supported single-qubit gates are RY and RZ, 2-qubit gates are CX, CZ and CRy.

        Only support the entanglers of "downstairs" structure.

    Args:
        n_qubits (int): The number of qubits.
        d (int): The depth of HE-ansatz.
        layers (list): The list of layers. Supported layers are "CX", "CZ", "CRy", "RY", "RZ".
        readout (list, optional): The readout qubits. Defaults to None.

    Examples:
        >>> from QuICT.algorithm.quantum_machine_learning.ansatz_library import HEAnsatz
        >>> ansatz = HEAnsatz(3, 1, ["RZ", "RY", "RZ", "CX"], [0])
        >>> circuit = ansatz.init_circuit()
        >>> circuit.draw("command")
                 ┌────────────┐┌────────────┐┌─────────────┐            ┌────┐
        q_0: |0>─┤ rz(1.0306) ├┤ ry(1.2941) ├┤ rz(-1.0592) ├──■─────────┤ cx ├
                ┌┴────────────┤├───────────┬┘├─────────────┤┌─┴──┐      └─┬──┘
        q_1: |0>┤ rz(-0.8802) ├┤ ry(1.346) ├─┤ rz(-1.0702) ├┤ cx ├──■─────┼───
                ├─────────────┤├───────────┴┐├─────────────┤└────┘┌─┴──┐  │
        q_2: |0>┤ rz(0.40935) ├┤ ry(1.1894) ├┤ rz(-2.4843) ├──────┤ cx ├──■───
                └─────────────┘└────────────┘└─────────────┘      └────┘
    """

    @property
    def readout(self):
        """Get the readout qubits.

        Returns:
            list: The list of readout qubits.
        """
        return self._readout

    def __init__(self, n_qubits: int, d: int, layers: list, readout: list = None):
        """Initialize an HE-ansatz instance."""
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

        Raises:
            AnsatzShapeError: An error occurred defining trainable parameters.
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
