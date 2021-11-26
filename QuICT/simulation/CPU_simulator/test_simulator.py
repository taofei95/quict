import numpy as np

from QuICT.core import *
from QuICT.algorithm import Amplitude


def test_sim():
    for qubit_num in range(2, 20):
        circuit = Circuit(qubit_num)
        circuit.random_append(20)
        res = Amplitude.run(circuit)  # New simulator would be used by default.
        expected = Amplitude.run(circuit, ancilla=None, use_old_simulator=True)
        flag = np.allclose(res, expected)
        assert flag
        # print(f"Testing for qubit {qubit_num}: {flag}")
