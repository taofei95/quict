import numpy as np
from mindquantum.core.circuit import Circuit, controlled
from mindquantum.core.gates import ZZ, XX
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.framework import MQAnsatzOnlyLayer
from mindquantum.simulator import Simulator


class HIQAnsatz:
    def __init__(self, n_qubits: int, color_qubit: int, readout: int, layers: int = 1):
        self._n_qubits = n_qubits
        readout = n_qubits - 1 if readout is None else readout
        color_qubit = n_qubits - 2 if color_qubit is None else color_qubit
        if (
            readout < 0
            or readout >= self._n_qubits
            or color_qubit < 0
            or color_qubit >= self._n_qubits
            or readout == color_qubit
        ):
            raise ValueError
        self._color_qubit = color_qubit
        self._readout = readout
        self._layers = layers

    def __call__(self):
        ansatz = Circuit()
        for l in range(self._layers):
            for k in range(self._n_qubits - 1, 1, -1):
                ansatz.xx(f"{l}-{k}xx", [k, self._readout])
                ansatz.xx(f"{l}-{k}xx", [k, self._color_qubit])
            for k in range(self._n_qubits - 1, 1, -1):
                ansatz.zz(f"{l}-{k}zz", [k, self._readout])
                ansatz.zz(f"{l}-{k}zz", [k, self._color_qubit])
        return ansatz
