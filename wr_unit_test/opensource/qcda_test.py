

import random
from QuICT.core.circuit.circuit import Circuit
from QuICT.core.gate import *
from QuICT.core.utils.gate_type import GateType
from QuICT.core.layout import Layout
from QuICT.algorithm.qft.quantum_frourier_transform import QFT, IQFT
from QuICT.core.virtual_machine.special_set.google_set import *
from QuICT.simulation.simulator import Simulator
from QuICT.simulation.state_vector.statevector_simulator import StateVectorSimulator
from QuICT.tools.interface.qasm_interface import OPENQASMInterface
from scipy.stats import unitary_group

from qiskit import Aer, qasm2, transpile, QuantumCircuit

cir = Circuit(5)
cir.random_append(10)
cir.draw("command")
for c in cir.gates:
    print(c)
    print(c.commutative(H & 0))


# lay_a = Layout(5)
# lay_a.add_edge(1, 2)
# lay_a.edge_list[0].u = 2
# # assert lay_a.edge_list[0].u != lay_a.edge_list[0].v

# print(cgate.qasm())

# circ = QuantumCircuit(5)
# circ.h(1)
# circ.h(0)
# circ.h(4)
# circ.h(2)
# circ.h(0)
# circ.h(2)
# circ.h(2)
# circ.h(4)
# circ.h(3)
# circ.h(0)
# circ.measure_all()

# cir = OPENQASMInterface.load_file("quict_qasm.txt").circuit
# sim = d_sim = Simulator(backend="density_matrix")
# DM = sim.run(cir)

# circ = qasm2.load(
#     filename="qiskit_qasm.txt",
#     include_path=qasm2.LEGACY_INCLUDE_PATH,
#     custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
#     custom_classical=qasm2.LEGACY_CUSTOM_CLASSICAL,
#     )

# # execute the quantum circuit
# backend = Aer.get_backend('aer_simulator_density_matrix')
# circ = transpile(circ, backend)
# circ.save_density_matrix()
# result = backend.run(circ).result()

# print("\n")
# print(result.data()["density_matrix"][1])
# print("\n")
# print(DM["data"]["density_matrix"])

# cir = Circuit(5)

# matrix = unitary_group.rvs(2 ** 2)
# u = Unitary(matrix)
# u | cir(random.sample(list(range(5)), 2))
# print(cir.qasm())
# cir.gate_decomposition()
# print(cir.qasm())

# cgate1 = CompositeGate()
# with cgate1:
#     for _ in range(1):
#         H & 0
#         CX & [2, 1]
#         CCRz & [3, 2, 1]
# cgate2 = CompositeGate()
# cgate3 = CompositeGate()
# with cgate2:
#     H & 0
#     CX & [1, 2]
#     CCRz & [1, 2, 3]
# for _ in range(6):
#     cgate2 | cgate3
#     cgate3 | cgate1
#     print(cgate1.size())
# cgate1 | cir
# print(cir.size(), 3 * 2 ** 6)

# cir.random_append(10)
# sim = StateVectorSimulator()
# sv = sim.run(cir)
# np.save("test.npy", sv)
# f1 = np.load("test.npy")
# print(f1)
# cir = Circuit(5)
# cir.random_append(10)
# quict_qasm = open("quict_qasm.txt", "w+")
# quict_qasm.write(cir.qasm())
# quict_qasm.close()

# with open("quict_qasm.txt", "r",) as f:
#     for line in f.readlines():
#         if len(line.split(")")) == 2:
#             l = str(line.split(")")[1])
#         else:
#             l = str(line.split(")"))
#         a = [i for i in l if str.isdigit(i)]


# for line in lines:
#     print("1")
#     print(line)

# qft_gate = QFT(3)
# qft_gate | cir([1,2,3])

# insert_index = random.choice(list(range(5)))
# cgate.insert(cgate, insert_index)
# # cgate | cir(0)
# cgate.draw("command")

# layout = Layout(5, "a")
# l = layout.linear_layout(10)
# print(l)
# c = CompositeGate()
# cir = Circuit(8)
# cir.random_append(30, [GateType.crz])
# depth = [0] * 10
# for gate in cir.gates:
#     print(gate)
#     if gate.controls + gate.targets == 1:
#         index = gate.targs + gate.cargs
#         print(index)
#         depth[index[0]] += 1
#     elif gate.controls + gate.targets == 2:
#         index = gate.targs + gate.cargs
#         new_depth = max(depth[index[0]], depth[index[1]]) + 1
#         depth[index[0]] = new_depth
#         depth[index[1]] = new_depth
#     elif gate.controls + gate.targets == 3:
#         index = gate.targs + gate.cargs
#         new_depth = max(
#             depth[index[0]],
#             depth[index[1]],
#             depth[index[2]]
#         ) + 1
#         depth[index[0]] = new_depth
#         depth[index[1]] = new_depth
#         depth[index[2]] = new_depth
# print(cir.depth(), np.max(depth))

# i_l = []
# d = [0] * 10
# for i in cir.gates:
#     d[i.targs[0]] += 1
# print(np.max(d), cir.depth())

# f = open("aa.txt", "w+")
# f.write(cir.qasm())
# print(cir.size() ,(8 * 10 + cir.count_gate_by_gatetype(GateType.fsim)))

# cir = Circuit(5)
# c1 = CompositeGate()
# H | c1(8)
# CCRz | c1([1, 2, 3])
# c1 | cir([2, 3, 5, 4])
# cir.draw("command")

# # gate with one qubit
# one_qubits_gate = [
#     GateType.h, GateType.s, GateType.sdg, GateType.x, GateType.y, GateType.z,
#     GateType.sx, GateType.sy, GateType.sw, GateType.id, GateType.u1, GateType.u2, GateType.u3,
#     GateType.rx, GateType.ry, GateType.rz, GateType.t, GateType.tdg, GateType.phase,
#     GateType.measure, GateType.reset, GateType.barrier
# ]
# # gate with two qubits
# two_qubits_gate = [
#     GateType.cx, GateType.cz, GateType.ch, GateType.crz, GateType.cu1, GateType.cu3, GateType.fsim,
#     GateType.rxx, GateType.ryy, GateType.rzz, GateType.swap, GateType.iswap
# ]
# # gate with three qubits
# three_qubits_gate = [GateType.ccx, GateType.ccz, GateType.cswap]
# # gate with params
# parameter_gates_for_call_test = [U1, U2, CU3, FSim]

# cir = Circuit(5)
# cir.random_append(50, one_qubits_gate+two_qubits_gate)
# f = open("qiskit_test_cir.qasm", "w+")
# f.write(cir.qasm())

# import qiskit.qasm2
# from qiskit import Aer, transpile
# from qiskit.qasm2 import LEGACY_CUSTOM_INSTRUCTIONS

# lay_b = Layout.linear_layout(5)
# print(lay_b)

# lay_a = Layout(10)
# lay_a.add_edge(1, 2)
# lay_a.add_edge(1, 3)
# lay_a.add_edge(2, 3)
# print(lay_a)
# print(lay_a.get_sublayout_edges([1, 2, 3]))
# print(lay_a.get_sublayout_edges([3, 2, 1]))


# err = [0.1, 0.2, 0.3, 0.4]
# gird_layout = Layout.grid_layout(4, width=2, error_rate=err)
# print(gird_layout)

# for i, edge in enumerate(gird_layout.edge_list):
#     assert edge.error_rate == err[i], \
#         f"the error rate of ({edge.u},{edge.v}) should be {err[i]}, but given {edge.error_rate} "

# circuit = qiskit.qasm2.load(
#     filename="qiskit_all_gates_circuit.qasm",
#     # include_path=qiskit.qasm2.LEGACY_INCLUDE_PATH,
#     # custom_instructions=qiskit.qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
#     custom_classical=qiskit.qasm2.LEGACY_CUSTOM_CLASSICAL,
#     )
# # circuit.draw()+
# simulator = Aer.get_backend('aer_simulator_statevector')
# circ = transpile(circuit, simulator)
# sv = simulator.run(circ)
# print(sv)

# from scipy.stats import unitary_group

# cir = Circuit(10)
# matrix = unitary_group.rvs(2 ** 2)
# u = Unitary(matrix)
# u | cir(random.sample(list(range(6)), 2))
# qft_gate = QFT(3)
# qft_gate | cir(random.sample(list(range(5)), 3))
# print(cir.qasm())

# mct = MultiControlToffoli(aux_usage='one_clean_aux')
# mct_gate = mct(3)
# mct_gate | cir
# print(cir.qasm())
# cir.random_append(10, [GateType.h, GateType.cx])
# sub_cir_gate = cir.sub_circuit(gate_limit=[GateType.cx])
# sub_cir_gate | cir
# print(cir.size())