# import sys
# sys.path.append('/mnt/e/ICT/QuICT')

import random

import numpy as np

from QuICT.algorithm import Amplitude, SyntheticalUnitary
from QuICT.core import *
from QuICT.qcda.synthesis.unitary_transform.uniformly_ry_revision import uniformlyRyRevision

def test_uniform_ry():
    for i in range(1, 8):
        circuit = Circuit(i)
        angles = [random.random() for _ in range(1 << (i - 1))]
        uniformlyRyRevision(angles) | circuit
        unitary = SyntheticalUnitary.run(circuit)
        for j in range(1 << (i - 1)):
            unitary_slice = unitary[2 * j:2 * (j + 1), 2 * j:2 * (j + 1)]
            circuit.print_information()
            assert not np.any(abs(unitary_slice - Ry(angles[j]).matrix.reshape(2, 2)) > 1e-10)

            
if __name__ == "__main__":
    test_uniform_ry()
