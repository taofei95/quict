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

    def __init__(
        self,
        n_qubits: int,
        d: int,
        layers: list,
        entangler: str = "downstair",
        readout: list = None,
    ):
        """Initialize an HE-ansatz instance.

        Args:
            n_qubits (int): The number of qubits.
            d (int): The depth of HE-ansatz.
            layers (list): The list of layers. Supported layers are "CX", "CZ", "CRy", "RY", "RZ".
            entangler (str): The type of entangler. Supported types are "downstair" and "full".Defaults to "downstair".
            readout (list, optional): The readout qubits. Defaults to None.
        """
        super(HEAnsatz, self).__init__(n_qubits)
        self._d = d
        self._layers = layers
        assert entangler in ["downstair", "full"]
        self._entangler = entangler
        self._n_params = 0
        self._readout = [0] if readout is None else readout
        self._validate_layers()

    def __str__(self):
        return "HEAnsatz(n_qubits={}, d={}, layers={}, entangler={})".format(
            self._n_qubits, self._d, self._layers, self._entangler
        )

    def init_circuit(self, params: Union[Variable, np.ndarray] = None):
        """Initialize an HE-ansatz with trainable parameters.

        Args:
            params (Union[Variable, np.ndarray], optional): Initialization parameters. Defaults to None.

        Returns:
            Circuit: The HE-ansatz ansatz.
        """
        params = np.random.randn(self._d, self._n_params) if params is None else params
        params = Variable(pargs=params) if isinstance(params, np.ndarray) else params
        if params.shape == (self._d, self._n_params):
            self._params = params
        else:
            raise AnsatzShapeError(str(self._d, self._n_params), str(params.shape))

        gate_dict = {"CX": CX, "CZ": CZ, "CRy": CRy, "RY": Ry, "RZ": Rz}
        circuit = Circuit(self._n_qubits)
        for i in range(self._d):
            param_id = 0
            for gate in self._layers:
                if gate in ["RY", "RZ"]:
                    for qid in range(self._n_qubits):
                        gate_dict[gate](params[i][param_id]) | circuit(qid)
                        param_id += 1
                elif gate in ["CRy"]:
                    if self._entangler == "downstair":
                        for qid in range(self._n_qubits - 1):
                            gate_dict[gate](params[i][param_id]) | circuit(
                                [qid, qid + 1]
                            )
                            param_id += 1
                        gate_dict[gate](params[i][param_id]) | circuit(
                            [self._n_qubits - 1, 0]
                        )
                        param_id += 1
                    else:
                        for qid1 in range(self._n_qubits - 1):
                            invert = False
                            for qid2 in range(qid1 + 1, self._n_qubits):
                                qubit_indexes = [qid2, qid1] if invert else [qid1, qid2]
                                gate_dict[gate](params[i][param_id]) | circuit(
                                    qubit_indexes
                                )
                                invert = not invert
                                param_id += 1
                else:
                    if self._entangler == "downstair":
                        for qid in range(self._n_qubits - 1):
                            gate_dict[gate] | circuit([qid, qid + 1])
                        gate_dict[gate] | circuit([self._n_qubits - 1, 0])
                    else:
                        for qid1 in range(self._n_qubits - 1):
                            invert = False
                            for qid2 in range(qid1 + 1, self._n_qubits):
                                qubit_indexes = [qid2, qid1] if invert else [qid1, qid2]
                                gate_dict[gate] | circuit(qubit_indexes)
                                invert = not invert

        return circuit

    def _validate_layers(self):
        for layer in self._layers:
            if layer in ["RY", "RZ"]:
                self._n_params += self._n_qubits
            elif layer in ["CRy"]:
                self._n_params += (
                    self._n_qubits
                    if self._entangler == "downstair"
                    else int(self._n_qubits * (self._n_qubits - 1) / 2)
                )
            elif layer not in ["CX", "CZ"]:
                raise AnsatzValueError('["RY", "RZ", "CRy", "CX", "CZ"]', layer)
