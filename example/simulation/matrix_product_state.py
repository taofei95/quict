import numpy as np
from scipy.stats import unitary_group
import random

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.simulation.state_vector import StateVectorSimulator
from QuICT.simulation.matrix_product_state import MatrixProductStateSimulator



def mps_test(qubits, repeat):
    single_gate = [GateType.h, GateType.x, GateType.rx, GateType.ry, GateType.u1, GateType.phase, GateType.z]
    double_gate = [GateType.swap, GateType.cx, GateType.cz, GateType.fsim, GateType.rxx, GateType.ryy, GateType.rzz, GateType.rzx]
    tri_gate = [GateType.ccx, GateType.cswap, GateType.ccz, GateType.ccrz]
    edges = [[i, i - 1] for i in range(1, qubits)] + [[i, i + 1] for i in range(qubits - 1)]

    sv_sim = StateVectorSimulator()
    sd = MatrixProductStateSimulator()

    for _ in range(repeat):
        cir = Circuit(qubits)
        sp = np.random.random(1 << qubits).astype(np.complex128)
        cir.random_append(10 * qubits, single_gate + double_gate)
        for _ in range(qubits):
            matrix = unitary_group.rvs(2 ** 2)
            curr_idx = random.sample(range(qubits), 2)
            Unitary(matrix) | cir(curr_idx)

            matrix = unitary_group.rvs(2 ** 1)
            curr_idx = np.random.randint(0, qubits)
            Unitary(matrix) | cir(curr_idx)

        cir.random_append(qubits, tri_gate)

        h = sv_sim.run(cir, sp)
        x = sd.run(cir, sp)
        xx = x.to_statevector()
        print(np.allclose(h, xx))


def single_gate_test():
    np.random.seed(2023)
    qubits = 5
    sv_sim = StateVectorSimulator()
    sd = MatrixProductStateSimulator()
    sp = np.random.random(1 << qubits).astype(np.complex128)

    cir = Circuit(qubits)
    cir.random_append(5, [GateType.ccx, GateType.cswap, GateType.ccz])
    # CCX | cir([2, 0, 3])
    # CSwap | cir([0, 1, 4])
    # CCZ | cir([3, 1, 0])
    # CSwap | cir([3, 2, 1])
    # CSwap | cir([4, 1, 3])
    # CCX | cir([2, 0, 3])
    # CSwap | cir([0, 1, 4])
    # CCZ | cir([2, 3, 4])
    # CCX | cir([2, 1, 0])
    # CCRz(np.pi) | cir([2, 1, 0])
    # CX | cir([2, 0])
    # for i in range(4):
    #     matrix = unitary_group.rvs(2 ** 1)
    #     Unitary(matrix) | cir(i)

    # for i in range(0, 3, 1):
    #     matrix = unitary_group.rvs(2 ** 2)
    #     Unitary(matrix) | cir([i, i+1])
    # gate_builder(GateType.cx, random_params=True) | cir([0, 1])

    x = sd.run(cir, sp)
    xx = x.to_statevector()

    h = sv_sim.run(cir, sp)
    # print(h)
    
    # print(xx)

    print(np.allclose(h, xx))
    print(cir.qasm())


def qsp_test(qubit):
    cir = Circuit(qubit)
    sp = np.random.random(1 << qubit).astype(np.complex128)

    sd = MatrixProductStateSimulator()
    x = sd.run(cir, sp)

    xx = x.to_statevector()
    assert np.allclose(xx, sp)


def cir_test():
    tri_gate = [GateType.ccx, GateType.cswap, GateType.ccz, GateType.ccrz]
    cir = Circuit(5)
    cir.random_append(10, tri_gate)

    for gate, qidx, _ in cir.fast_gates:
        print(gate.cargs + gate.targs)
        print(qidx)
        cgate = gate.build_gate()
        print(cgate.qubits)
        print('---------------------')

    print(cir.qasm())


if __name__=="__main__":
    mps_test(10, 10)
    # for _ in range(1):
    #     single_gate_test()
    #     qsp_test(15)
