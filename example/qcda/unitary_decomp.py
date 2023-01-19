
from scipy.stats import unitary_group

from QuICT.core import Circuit
from QuICT.qcda.synthesis.unitary_decomposition import UnitaryDecomposition


if __name__ == '__main__':
    U = unitary_group.rvs(2 ** 3)
    UD = UnitaryDecomposition()
    compositeGate, _ = UD.execute(U)

    circuit = Circuit(3)
    circuit.extend(compositeGate)
    circuit.draw(filename="unitary_decomposition")
