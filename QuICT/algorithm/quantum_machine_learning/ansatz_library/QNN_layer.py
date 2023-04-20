import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import *


class QNNLayer:
    """Initialize a QNNLayer instance."""

    __DEVICE = ["CPU", "GPU"]

    def __init__(self, n_qubits: int, readout: int):
        """The QNN layer constructor.
        """
        self._n_qubits = n_qubits
        if readout < 0 or readout >= self._n_qubits:
            raise ValueError
        self._data_qubits = list(range(n_qubits)).remove(readout)
        self._readout = readout

    def __call__(self, two_qubit_gates, params: Variable):
        if not isinstance(two_qubit_gates, list):
            two_qubit_gates = [two_qubit_gates]
        n_layers = len(two_qubit_gates)
        if params.shape[0] != n_layers or params.shape[1] != self._n_qubits - 1:
            raise ValueError

        gate_dict = {"XX": Rxx, "YY": Ryy, "ZZ": Rzz, "ZX": Rzx}
        circuit = Circuit(self._n_qubits)
        for l, gate in zip(range(n_layers), two_qubit_gates):
            if gate not in gate_dict.keys():
                raise ValueError

            for i in range(self._n_qubits - 1):
                gate_dict[gate](params[l][i]) | circuit(
                    [self._data_qubits[i], self._readout]
                )
        return circuit
