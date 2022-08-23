import numpy as np

from QuICT.algorithm import SyntheticalUnitary
from QuICT.core import Circuit
from QuICT.core.gate import Ry
from QuICT.qcda.synthesis.unitary_decomposition.uniformly_ry_revision import UniformlyRyRevision


def test_uniform_ry():
    for _ in range(10):
        for i in range(1, 6):
            circuit = Circuit(i)
            angles = [np.random.uniform(low=0, high=np.pi) for _ in range(1 << (i - 1))]
            URyRevision = UniformlyRyRevision()
            URyRevision.execute(angles) | circuit
            unitary = SyntheticalUnitary.run(circuit)
            # print(unitary)
            for j in range(1 << (i - 1)):
                unitary_slice = unitary[2 * j:2 * (j + 1), 2 * j:2 * (j + 1)]
                assert np.allclose(unitary_slice, Ry(angles[j]).matrix)
