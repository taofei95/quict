import os
import unittest
import numpy as np
import cupy as cp

from QuICT.core import Circuit
from QuICT.core.gate import H, S
from QuICT.simulation.utils import GateMatrixs


# @unittest.skipUnless(os.environ.get("test_with_gpu", False), "require GPU")
class TestGateMatrix(unittest.TestCase):
    def test_gate_matrix(self):
        # example circuit
        circuit = Circuit(2)
        H | circuit
        S | circuit

        gate_matrix = GateMatrixs(np.complex128)
        gate_matrix.build(circuit.gates)
        gmatrix = gate_matrix.get_target_matrix(circuit.gates[0])

        self.assertTrue(np.allclose(gmatrix, circuit.gates[0].matrix.ravel()))


if __name__ == "__main__":
    unittest.main()
