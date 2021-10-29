import sys

import numpy as np

from QuICT.core import *
from .port import run_simulation, sim_back_bind
from QuICT.algorithm import Amplitude


def test_sim():
    print()
    for qubit_num in range(2, 20):
        circuit = Circuit(qubit_num)
        circuit.random_append(20)
        res = run_simulation(circuit)
        expected = Amplitude.run(circuit)
        assert np.allclose(res, expected)
