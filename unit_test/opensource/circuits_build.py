import string
import random
from QuICT.algorithm.qft.quantum_frourier_transform import QFT, IQFT
from QuICT.core.circuit.circuit import Circuit
from QuICT.core.gate import *
from QuICT.core.utils.gate_type import GateType
from scipy.stats import unitary_group

# gate with one qubit
one_qubits_gate = [
    GateType.h, GateType.hy, GateType.s, GateType.sdg, GateType.x, GateType.y, GateType.z,
    GateType.sx, GateType.sy, GateType.sw, GateType.id, GateType.u1, GateType.u2, GateType.u3,
    GateType.rx, GateType.ry, GateType.rz, GateType.t, GateType.tdg, GateType.phase, GateType.gphase,
    GateType.measure, GateType.reset, GateType.barrier
]
# gate with two qubits
two_qubits_gate = [
    GateType.cx, GateType.cz, GateType.ch, GateType.crz, GateType.cu1, GateType.cu3, GateType.fsim,
    GateType.rxx, GateType.ryy, GateType.rzz, GateType.swap, GateType.iswap, GateType.iswapdg, GateType.sqiswap
]
# gate with three qubits
three_qubits_gate = [GateType.ccx, GateType.ccz, GateType.ccrz, GateType.cswap]
# gate with params
parameter_gates_for_call_test = [U1, U2, CU3, FSim, CCRz]

for _ in range(1):
    # the qubit and the probility of gate, and cir size is qubit * gate_prob
    qubits_number = list(range(5, 15))
    gates_prob = list(range(10, 50, 5))

    # random choice qubit and gate
    q = random.choice(qubits_number)
    g_prob = random.choice(gates_prob)

    # inital quantum circuit
    cir = Circuit(q)
    # ----------------------------
    # random append gate to circuit
    # ----------------------------
    cir.random_append(q*g_prob, one_qubits_gate+two_qubits_gate, random_params=True)

    # ----------------------------
    # append supremacy to circuit
    # ----------------------------
    cir.supremacy_append(3)

    # ----------------------------
    # QFT IQFT Unitary Perm PermFx MultiControlGate MultiControlToffoli
    # ----------------------------
    qft_gate = QFT(3)
    qft_gate | cir(random.sample(qubits_number, 3))

    iqft_gate = IQFT(3)
    iqft_gate | cir(random.sample(qubits_number, 3))

    matrix = unitary_group.rvs(2 ** 2)
    u = Unitary(matrix)
    u | cir(random.sample(qubits_number, 2))

    # TODO: bug
    # mc = MultiControlGate(controls=1, gate_type=GateType.h, params=[0, 0])
    # mc | cir([0, 1])

    mct = MultiControlToffoli(aux_usage='one_clean_aux')
    mct_gate = mct(3)
    mct_gate | cir

    # ----------------------------
    # build sub circuit
    # ----------------------------
    sub_cir_qubit = cir.sub_circuit(qubit_limit=[0, 1, 2])
    sub_cir_qubit | cir
    sub_cir_size = cir.sub_circuit(max_size=20)
    sub_cir_size | cir
    sub_cir_gate = cir.sub_circuit(gate_limit=[GateType.cx])
    sub_cir_gate | cir

    # ----------------------------
    # add qubit to cir
    # ----------------------------
    cir.add_qubit(5, is_ancillary_qubit=False)
    cir.add_qubit(5, is_ancillary_qubit=True)

    # ----------------------------
    # circuit
    # ----------------------------
    for _ in range(5):
        qubit_range = list(range(q))
        gate = random.choice(one_qubits_gate+two_qubits_gate+three_qubits_gate)
        gate = gate_builder(gate, random_params=True)
        if gate.controls + gate.targets == 1:
            index = [random.choice(qubit_range)]
        elif gate.controls + gate.targets == 2:
            index = list(random.sample(qubit_range, 2))
        elif gate.controls + gate.targets == 3:
            index = list(random.sample(qubit_range, 3))

        rand_j = random.randint(0, 6)
        if rand_j == 0:
            gate | cir(index)
        elif rand_j == 1:
            gate & index | cir
        elif rand_j == 2:
            cir.append(gate & index)
        elif rand_j == 3:
            insert_index = random.choice(qubit_range)
            cir.insert(gate & index, insert_index)
        elif rand_j == 4:
            cir.extend(cir)
        elif rand_j == 5:
            cir | cir
        elif rand_j == 6:
            for _ in range(5):
                cir | cir
        elif rand_j == 7:
            for _ in range(3):
                cir | cir(list(range(cir.width())))

    # ----------------------------
    # compositegate
    # ----------------------------
    c = CompositeGate()
    c1 = CompositeGate()

    for _ in range(10):
        qubit_range = list(range(q))
        gate = random.choice(one_qubits_gate+two_qubits_gate+three_qubits_gate)
        gate = gate_builder(gate, random_params=True)
        if gate.controls + gate.targets == 1:
            index = [random.choice(qubit_range)]
        elif gate.controls + gate.targets == 2:
            index = list(random.sample(qubit_range, 2))
        elif gate.controls + gate.targets == 3:
            index = list(random.sample(qubit_range, 3))

        rand_j = random.randint(0, 8)
        if rand_j == 0:
            print("append gate begin")
            gate | c(index)
            gate & index | c
            c.append(gate & index)
        elif rand_j == 3:
            print("insert gate begin")
            insert_index = random.choice(qubit_range)
            c.insert(gate & index, insert_index)
            c.insert(c, insert_index)
        elif rand_j == 4:
            print("append cgate to cgate begin")
            c | c1
            c1 | c
            c | c
            c1 | c1
        elif rand_j == 5:
            print("extend gate begin")
            # TODO: bug
            # c.extend(c)
        elif rand_j == 7:
            print("special append gate begin")
            with c:
                for _ in range(5):
                    gate & index

    # ----------------------------
    # inverse cir
    # ----------------------------
    cir.inverse() | cir

    # ----------------------------
    # gate decomposition
    # ----------------------------
    cir_decomposition = cir.gate_decomposition()
    c_decomposition = c.gate_decomposition()

    print(c.qasm())
    print(cir.qasm())
    cir.draw("command")

