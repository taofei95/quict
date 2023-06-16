import numpy as np
from mindquantum.core.circuit import Circuit, controlled
from mindquantum.core.gates import ZZ, XX
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.framework import MQAnsatzOnlyLayer
from mindquantum.simulator import Simulator


class HIQAnsatz:
    def __init__(self, n_qubits: int, color_qubit: int = None, readout: int = None):
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
        self._data_qubits = list(range(n_qubits))
        self._data_qubits.remove(readout)
        self._data_qubits.remove(color_qubit)
        self._color_qubit = color_qubit
        self._readout = readout

    def __call__(self):
        circuit = Circuit()
        for i in self._data_qubits:
            circuit += XX(f"xr{i}").on([i, self._readout])
            circuit += XX(f"xc{i}").on([i, self._color_qubit])
        for i in self._data_qubits:
            circuit += ZZ(f"zr{i}").on([i, self._readout])
            circuit += ZZ(f"zc{i}").on([i, self._color_qubit])

        return circuit
