import os
import random
from QuICT.algorithm.qft.quantum_frourier_transform import QFT, IQFT
from QuICT.core.circuit.circuit import Circuit
from QuICT.core.gate import *
from QuICT.core.utils.gate_type import GateType
from scipy.stats import unitary_group
from QuICT.core.virtual_machine.instruction_set import InstructionSet
from QuICT.qcda.synthesis.gate_transform.gate_transform import GateTransform
from qiskit import qasm2
from QuICT.tools.interface.qasm_interface import OPENQASMInterface


all_gates = [
    GateType.h, GateType.s, GateType.sdg, GateType.x, GateType.y, GateType.z,
    GateType.id, GateType.u1, GateType.u2, GateType.u3, GateType.rx, GateType.ry,
    GateType.rz, GateType.t, GateType.tdg, GateType.cx, GateType.cz, GateType.cu1,
    GateType.cu3, GateType.rxx, GateType.swap, GateType.ccx, GateType.cswap
]

# test circuit infos with qiskit
def all_gates_random_circuit(q):
    cir_quict = Circuit(q)
    cir_qiskit = Circuit(q)
    qubit_range = list(range(q))
    for gate in all_gates:
        gate = gate_builder(gate, random_params=True)
        gate_index = gate.controls + gate.targets
        index_quict = list(random.sample(qubit_range, gate_index))
        index_qiskit = []
        for i in range(1, gate_index + 1):
            index_qiskit.append(q - 1 - index_quict[i - 1])

        gate & index_quict | cir_quict
        gate & index_qiskit | cir_qiskit
    return cir_quict, cir_qiskit

def special_random_circuit(q):
    cir_quict = Circuit(q)
    cir_qiskit = Circuit(q)
    cgate1, cgate2 = all_gates_random_circuit(q)
    cgate1.to_compositegate()
    cgate2.to_compositegate()

    for _ in range(3):
        cgate1 | cgate1
    cgate1 | cir_quict
    for _ in range(3):
        cgate2 | cgate2
    cgate2 | cir_qiskit
    return cir_quict, cir_qiskit

def test_cir_infos_with_qiskit(q):
    # cir_quict, cir_qiskit = all_gates_random_circuit(q)
    cir_quict, cir_qiskit = special_random_circuit(q)

    f = open("inverse.qasm", "w+")
    f.write(cir_qiskit.qasm())
    f.close()
    cir_qiskit = qasm2.load(
        filename="inverse.qasm",
        include_path=qasm2.LEGACY_INCLUDE_PATH,
        custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
        custom_classical=qasm2.LEGACY_CUSTOM_CLASSICAL,
    )
    print(cir_quict.width(), cir_qiskit.width())
    print(cir_quict.size(), cir_qiskit.size())
    print(cir_quict.depth(), cir_qiskit.depth())

def test_random_supremacy_sub():
    # inital quantum circuit
    q = random.choice(list(range(5, 15)))
    g_prob = random.choice(list(range(5, 20, 5)))
    cir = Circuit(q)

    cir.random_append(rand_size=q*g_prob, typelist=all_gates, random_params=True, seed=10)

    cir.supremacy_append(1, "ABA")

    sub_cir_size = cir.sub_circuit(start=5)
    sub_cir_size | cir
    
def test_special_gate():
    # inital quantum circuit
    q = random.choice(list(range(5, 15)))
    cir = Circuit(q)

    qft_gate = QFT(3)
    qft_gate | cir(random.sample(list(range(q)), 3))

    iqft_gate = IQFT(3)
    iqft_gate | cir(random.sample(list(range(q)), 3))

    matrix = unitary_group.rvs(2 ** 2)
    # matrix = CX.matrix
    u = Unitary(matrix=matrix)
    u | cir(random.sample(list(range(q)), 2))

    mct = MultiControlToffoli(aux_usage='no_aux')
    mct_gate = mct(3)
    mct_gate | cir
    print(cir.qasm())

def test_gate_to_cir():
    q = random.choice(list(range(5, 15)))
    cir = Circuit(q)

    for _ in range(50):
        cgate = CompositeGate()
        qubit_range = list(range(q))
        gate = random.choice(all_gates)
        gate = gate_builder(gate, random_params=True)
        if gate.controls + gate.targets == 1:
            index = [random.choice(qubit_range)]
        elif gate.controls + gate.targets == 2:
            index = list(random.sample(qubit_range, 2))
        elif gate.controls + gate.targets == 3:
            index = list(random.sample(qubit_range, 3))

        rand_i = random.randint(0, 4)
        if rand_i == 0:
            gate | cir(index)
            gate | cgate(index)
            cgate | cir(index)
        elif rand_i == 1:
            gate & index | cir
            gate & index | cgate
            cgate | cir(index)
        elif rand_i == 2:
            cir.append(gate & index)
            cgate.append(gate & index)
            cgate | cir(index)
        elif rand_i == 3:
            insert_index = random.choice(qubit_range)
            cir.insert(gate & index, insert_index)
            cgate.insert(gate & index, insert_index)
            cgate | cir(index)
        elif rand_i == 4:
            with cgate:
                for _ in range(2):
                    gate & index
            cgate | cir
    print(cir.qasm())

def test_cir_to_cir():
    q = random.choice(list(range(5, 15)))
    g_prob = random.choice(list(range(5, 20, 5)))
    cir = Circuit(q)

    for _ in range(10):
        cgate1 = CompositeGate()
        for _ in range(5):
            qubit_range = list(range(q))
            gate = gate_builder(random.choice(all_gates[:10]), random_params=True)
            index = [random.choice(qubit_range)]
            with cgate1:
                gate & index

        cir1 = Circuit(q)
        cir1.random_append(rand_size=g_prob, typelist=all_gates)

        rand_j = random.randint(0, 6)
        if rand_j == 0:
            cir1.extend(cir1)
            cir.extend(cir1)
        elif rand_j == 1:
            for _ in range(q):
                cgate1.extend(cgate1)
            cir.extend(cgate1)
        elif rand_j == 2:
            for _ in range(q):
                cir1 | cir
        elif rand_j == 3:
            cir1 | cir1
            cir1 | cir
        elif rand_j == 4:
            for _ in range(q):
                cgate1 | cgate1
            cgate1 | cir
        elif rand_j == 5:
            cgate2 = CompositeGate()
            cgate3 = CompositeGate()
            with cgate2:
                H & 0
                CX & [1, 2]
                CCRz & [1, 2, 3]
            for _ in range(q):
                cgate2 | cgate3
                cgate3 | cgate1
            cgate1 | cir
        elif rand_j == 6:
            cir.inverse() | cir
    print(cir.qasm())

test_cir_to_cir()