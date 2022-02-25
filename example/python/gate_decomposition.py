
import numpy as np
from scipy.stats import unitary_group

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.qcda.synthesis.gate_decomposition import GateDecomposition


if __name__ == "__main__":
    circuit = Circuit(3)
    CX | circuit([0, 2])
    CCRz(np.pi / 3) | circuit([0, 1, 2])
    CSwap | circuit([0, 1, 2])
    matrix = unitary_group.rvs(2 ** 3)
    Unitary(matrix) | circuit([0, 1, 2])
    circuit.draw()

    gates_decomposed = GateDecomposition.execute(circuit)
    circuit_decomposed = Circuit(3)
    circuit_decomposed.set_exec_gates(gates_decomposed)
    circuit_decomposed.draw()
