import numpy as np
from QuICT.core import Circuit
from QuICT.core.gate import FSim, Measure, build_gate, SX, SY, H, CompositeGate
from QuICT.core.utils import GateType

def supr_circuit_build():
    supremacy_typelist = [GateType.sx, GateType.sy, GateType.sw]
    Agate_indexeslist = [[1, 2], [5, 6], [9, 10], [13, 14]]
    Bgate_indexeslist = [[1, 4], [5, 8], [9, 12], [3, 6], [7, 10], [11, 14]]
    Cgate_indexeslist = [[0, 1], [4, 5], [8, 9], [12, 13], [2, 3], [6, 7], [10, 11], [14, 15]]
    Dgate_indexeslist = [[1, 6], [5, 10], [9, 14]]

    ranks = ['A', 'B', 'C', 'D', 'C', 'D', 'A', 'B']
    cir = Circuit(16)

    H | cir
    for i in range(8):
        for q in range(16):
            gate_type = supremacy_typelist[np.random.randint(0, 3)]
            cir.append(build_gate(gate_type, q))

        num = ranks[i]
        if num == 'A':
            for qubits in Agate_indexeslist:
                FSim(np.pi / 2, np.pi / 6) | cir(qubits)
        if num == 'B':
            for qubits in Bgate_indexeslist:
                FSim(np.pi / 2, np.pi / 6) | cir(qubits)
        if num == 'C':
            for qubits in Cgate_indexeslist:
                FSim(np.pi / 2, np.pi / 6) | cir(qubits)
        if num == 'D':
            for qubits in Dgate_indexeslist:
                FSim(np.pi / 2, np.pi / 6) | cir(qubits)

    Measure | cir
    cir.draw()

supr_circuit_build()
