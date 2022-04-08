
from scipy.stats import unitary_group

from QuICT.core import Circuit
from QuICT.qcda.synthesis.unitary_transform import UnitaryTransform


if __name__ == '__main__':
    U = unitary_group.rvs(2 ** 3)
    compositeGate, _ = UnitaryTransform.execute(U)

    circuit = Circuit(3)
    circuit.extend(compositeGate)
    circuit.draw()
