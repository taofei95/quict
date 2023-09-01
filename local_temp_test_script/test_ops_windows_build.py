import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.core.operator import Trigger
from QuICT.core.utils import GateType


# Be aware that too many types at the same time may not benefit to the test,
# unless the size of the random circuit is also large.
typelist = [
    GateType.rx, GateType.ry, GateType.rz, GateType.x,
    GateType.y, GateType.z, GateType.cx
]


def cir_build():
    cir = Circuit(8)
    CX | cir([6,7])
    CX | cir([7,5])
    CX | cir([6,7])
    CCX | cir([7,6,5])
    CX | cir([6,7])
    CX | cir([1,4])
    CX | cir([6,3])
    CX | cir([3,4])
    CX | cir([4,5])
    CX | cir([0, 5])
    Z | cir(3)
    X | cir(4)
    CX | cir([4,3])
    CX | cir([3,1])
    X | cir(4)
    CX | cir([1,2])    
    CX | cir([3,1])
    CX | cir([3,5])
    CX | cir([3,6])
    X | cir(3)
    CX | cir([4,5])

    return cir


def sp1():
    circuit = Circuit(5)
    Rx | circuit(1)
    Rxx | circuit([2, 3])
    CRz | circuit([2, 0])
    Rxx | circuit([4,2])
    CRz | circuit([0,3])
    CRz | circuit([4,0])
    CY | circuit([3,2])
    Ry | circuit(2)
    CX | circuit([2,0])
    CH | circuit([2,1])
    CZ | circuit([3,0])
    Rzz | circuit([4,0])
    FSim | circuit([0, 1])
    CZ | circuit([4,2])
    Rx | circuit(2)
    CRz | circuit([4,3])
    Rxx | circuit([2,1])
    Ryy | circuit([3,2])
    Ryy | circuit([1,3])
    FSim | circuit([0,2])
    
    return circuit


def sp2():
    c = Circuit(5)
    with CompositeGate() as cg:
        T_dagger & 4
        Z & 1
        T_dagger & 3
        S & 0
        S & 4
        S & 0
        S & 1
        X & 3
        Y & 4
        H & 1
        CX & [2, 3]
        ID & 3
        T_dagger & 1
        S_dagger & 4
        CCX & [1, 3, 4]
        CCX & [0, 3, 2]
        S & 3
        T & 2
        ID & 3
        ID & 4

    cg | c

    # circuit.draw()

    return c


def convert_precision_test():
    circuit = Circuit(5)
    circuit.random_append(20)

    circuit.convert_precision()
    print(circuit.matrix().dtype)


def unitary_gate_build_test():
    from QuICT.qcda.synthesis.unitary_decomposition import UnitaryDecomposition
    from scipy.stats import unitary_group
    import random

    matrix = unitary_group.rvs(2 ** 3)
    target = random.sample(range(5), 3)
    print(target)
    ugate = Unitary(matrix) & target
    cgate1 = ugate.build_gate()
    print(cgate1.qasm())


def decomp_gate_test():
    from scipy.stats import unitary_group
    import random

    from QuICT.qcda.synthesis import GateDecomposition
    
    cir = Circuit(5)
    matrix = unitary_group.rvs(2 ** 5)
    target = random.sample(range(5), 5)
    Unitary(matrix) | cir(target)
    CCX | cir([0, 1, 3])
    QFT(3) | cir([2, 0, 1])
    CSwap | cir([4,2,3])

    print(cir.matrix())

    # GateDecomposition
    opt_circuit = GateDecomposition.execute(cir)
    print(opt_circuit)
    # print(opt_circuit.qasm())
    
    cir.gate_decomposition()
    print(cir)
    # print(cir.qasm())
    
    for i in range(cir.size()):
        g1 = opt_circuit.gates[i]
        g2 = cir.gates[i]
        assert g1.type == g2.type
        assert g1.cargs == g2.cargs
        if not (g1.targs == g2.targs):
            print(g1)
            print(g2)


def test_lastcall():
    cir = Circuit(5)
    cir.random_append(50)
    
    print(cir.qasm())
    print(cir.get_lastcall_for_each_qubits())


def test_sv_simu():
    from QuICT.simulation.state_vector import StateVectorSimulator
    from QuICT.simulation.density_matrix import DensityMatrixSimulation
    from QuICT.simulation.unitary import UnitarySimulator
    
    cir = Circuit(4)
    H | cir(0)

    cgate = CompositeGate()
    CX | cgate([0, 1])
    CX | cgate([1, 2])
    # CX | cgate([2, 3])

    # Build Trigger
    cgate | cir

    sim = StateVectorSimulator()
    sv = sim.run(cir)
    print(sv)
    print(sim.sample(100))
    sim_s = StateVectorSimulator("GPU")
    dm = sim_s.run(cir)
    print(sim_s.sample(100))


def test_parameterize():
    from QuICT.qcda.optimization.commutative_optimization import CommutativeOptimization

    test_list = [X, Y, Z, SX, SY, S, S_dagger, T, T_dagger]
    for gate in test_list:
        gate = gate & 0
        gate_para, phase = CommutativeOptimization.parameterize(gate)
        state = np.allclose(gate.matrix, np.exp(1j * phase) * gate_para.matrix)
        if not state:
            print(gate.type)
            print(gate.matrix)
            print(np.exp(1j * phase) * gate_para.matrix)


def test_deparameterize():
    from QuICT.qcda.optimization.commutative_optimization import CommutativeOptimization

    # Rx
    for k in range(8):
        gate = Rx(k * np.pi / 2) & 0
        gates_depara, phase = CommutativeOptimization.deparameterize(gate)
        state = np.allclose(gate.matrix, np.exp(1j * phase) * gates_depara.matrix())
        if not state:
            print(k)
            print(gate.matrix)
            print(np.exp(1j * phase) * gates_depara.matrix())


def test_draw():
    cir = Circuit(4)
    cir.random_append(20)
    cir.draw()


def test_circuitlib():
    from QuICT.lib import CircuitLib
    
    cl = CircuitLib()
    cirlist = cl.get_circuit("random", "qft", [5,13, 14])
    print(len(cirlist))
    for c in cirlist:
        print(c)


def test_vector_dot_matrix():
    from QuICT.ops.linalg.cpu_calculator import matrix_dot_vector, MatrixPermutation
    import random

    qubits = 10
    vector = np.random.random(1 << qubits).astype(np.complex128)
    matrix = np.random.random((1 << qubits, 1 << qubits)).astype(np.complex128)
    q_idx = list(range(qubits))
    random.shuffle(q_idx)

    np_matrix = MatrixPermutation(matrix, np.array(q_idx))
    np_result = np.dot(np_matrix, vector)
    quict_result = matrix_dot_vector(
        vector, qubits, matrix, q_idx
    )

    print(np.allclose(np_result, quict_result))
    print(np.sum(np_result), np.sum(quict_result))


if __name__ == '__main__':
    test_sv_simu()
