import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import *


class Qubit:
    """Qubit encoding for encoding classical image data into quantum circuits."""

    def __init__(self, data_qubits: int):
        self._data_qubits = data_qubits

    def encoding(self, img):
        img = img.flatten()
        assert img.shape[0] == self._data_qubits
        circuit = Circuit(self._data_qubits)
        for i in range(img.shape[0]):
            if img[i] == 1:
                X | circuit(i)
        return circuit
