import numpy as np
from scipy.stats import unitary_group

from QuICT.core.gate import UniformlyControlGate, GateType, Ry, Rz


def test():
    for target in [GateType.ry, GateType.rz, GateType.unitary]:
        ucg = UniformlyControlGate(target)
        for n in range(2, 4):
            if target in [GateType.ry, GateType.rz]:
                angles = [2 * np.pi * np.random.random() for _ in range(1 << (n - 1))]
                gates = ucg(angles)
                unitary = gates.matrix()
                for j in range(1 << (n - 1)):
                    unitary_slice = unitary[2 * j:2 * (j + 1), 2 * j:2 * (j + 1)]
                    if target == GateType.ry:
                        assert np.allclose(unitary_slice, Ry(angles[j]).matrix)
                    if target == GateType.rz:
                        assert np.allclose(unitary_slice, Rz(angles[j]).matrix)
            else:
                unitaries = [unitary_group.rvs(2) for _ in range(1 << (n - 1))]
                gates = ucg(unitaries)
                unitary = gates.matrix()
                if abs(unitary[0, 0]) > 1e-10:
                    delta = unitaries[0][0][0] / unitary[0, 0]
                else:
                    delta = unitaries[0][0][1] / unitary[0, 1]
                for j in range(1 << (n - 1)):
                    unitary_slice = unitary[2 * j:2 * (j + 1), 2 * j:2 * (j + 1)]
                    unitary_slice[:] *= delta
                    assert np.allclose(unitary_slice, unitaries[j].reshape(2, 2))
