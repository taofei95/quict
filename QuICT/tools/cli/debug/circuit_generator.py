from QuICT.core import Circuit
from QuICT.core.gate import *


normal_matrix_gates = [
    GateType.h, GateType.ch, GateType.hy, GateType.sx, GateType.sy, GateType.sw,
    GateType.u2, GateType.u3, GateType.rx, GateType.ry, GateType.cry, GateType.cu3
]
diagonal_control_matrix_gates = [
    GateType.crz, GateType.rzz, GateType.ccrz, GateType.rz, GateType.gphase,
    GateType.s, GateType.sdg, GateType.z, GateType.u1, GateType.t, GateType.swap,
    GateType.tdg, GateType.phase, GateType.cz, GateType.cu1, GateType.ccz,
]
swap_matrix_gate = [GateType.x, GateType.swap, GateType.cswap, GateType.iswap]
reverse_matrix_gate = [GateType.y, GateType.cx, GateType.cy, GateType.ccx]
other_matrix_gate = [GateType.fsim, GateType.sqiswap, GateType.rxx, GateType.ryy, GateType.rzx]
special_gate = [GateType.measure, GateType.reset, GateType.barrier]

single_gate = [
    GateType.h, GateType.hy, GateType.s, GateType.sdg, GateType.x, GateType.y,
    GateType.z, GateType.sx, GateType.sy, GateType.sw, GateType.id, GateType.u1,
    GateType.u2, GateType.u3, GateType.rx, GateType.ry, GateType.rz, GateType.t,
    GateType.tdg, GateType.phase, GateType.gphase
]
double_gate = [
    GateType.cz, GateType.cx, GateType.cy, GateType.ch, GateType.cry,
    GateType.crz, GateType.cu1, GateType.cu3, GateType.fsim, GateType.rxx,
    GateType.ryy, GateType.rzz, GateType.rzx,
    GateType.swap, GateType.iswap, GateType.iswapdg, GateType.sqiswap
]
tri_gate = [GateType.ccx, GateType.ccz, GateType.ccrz, GateType.cswap]


type_circuit_mapping = {
    "single": single_gate,
    "double": double_gate,
    "triple": tri_gate,
    "normal": normal_matrix_gates,
    "diagonal": diagonal_control_matrix_gates,
    "swap": swap_matrix_gate,
    "reverse": reverse_matrix_gate,
    "other": other_matrix_gate,
    "special": special_gate
}


def generate_random_circuit_by_type(
    type_: str,
    qubits: int = 10,
    seed: int = 2023
):
    circuit = Circuit(qubits)
    type_list = type_circuit_mapping[type_]
    circuit.random_append(
        len(type_list) * qubits, type_list, random_params=True, seed=seed
    )

    return circuit
