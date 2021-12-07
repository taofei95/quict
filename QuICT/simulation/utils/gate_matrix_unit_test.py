import os
import unittest
import numpy as np
import cupy as cp

from QuICT.core import *
from QuICT.simulation.utils import GateMatrixs


@unittest.skipUnless(os.environ.get("test_with_gpu", False), "require GPU")
class TestGateMatrix(unittest.TestCase):
    def test_gate_matrix(self):
        # example circuit
        circuit = Circuit(2)
        H | circuit
        S | circuit

        gate_matrix = GateMatrixs(np.complex128)
        for gate in circuit.gates:
            gate_matrix.build(gate)
        gate_matrix.concentrate_gate_matrixs()

        gate_matrix = gate_matrix.target_matrix(circuit.gates[0])

        self.assertTrue(np.allclose(gate_matrix, circuit.gates[0].compute_matrix.ravel()))


if __name__ == "__main__":
    unittest.main()
