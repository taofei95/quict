from yaml import add_implicit_resolver
from QuICT.core.gate.gate import *
from QuICT.core.circuit.circuit import Circuit
from QuICT.core.gate.gate_builder import build_gate
from QuICT.core.utils import gate_type


def build_sim_circuit():
    sim_typelist_1 = [GateType.h,GateType.x,GateType.y,GateType.z,GateType.phase,GateType.id]
    sim_typelist_2 = [GateType.s,GateType.t,GateType.sx,GateType.tdg,GateType.sdg]
    sim_typelist_3 = [GateType.rx,GateType.ry,GateType.rz,GateType.u1,GateType.u2,GateType.u3]
    
    circuit = Circuit(10)
    for i in range(5):
        for q in range(10):
            gate_type = sim_typelist_1[np.random.randint(0,5)]
            circuit.append(build_gate(gate_type,q))
    CX | circuit([0, 1])
    CX | circuit([1, 0])
    CY | circuit([2, 3])
    CY | circuit([3, 2])
    CZ | circuit([4, 5])
    CZ | circuit([5, 4])
    CH | circuit([6, 7])
    CH | circuit([7, 6])
    CRz | circuit([8, 9])
    CRz | circuit([9, 8])

    for i in range(5):
        for q in range(10):
            gate_type = sim_typelist_2[np.random.randint(0,5)]
            circuit.append(build_gate(gate_type,q))
    Rxx(np.pi) | circuit([0, 1])
    Rxx(np.pi) | circuit([1, 0])
    Ryy(np.pi) | circuit([2, 3])
    Ryy(np.pi) | circuit([3, 2])
    Rzz(np.pi) | circuit([4, 5])
    Rzz(np.pi) | circuit([5, 4])
    CU1(np.pi / 2) | circuit([6, 7])
    CU1(np.pi / 2) | circuit([7, 6])
    CU3(np.pi, 0, 1) | circuit([8, 9])
    CU3(np.pi, 1, 0) | circuit([9, 8])
    for i in range(5):
        for q in range(10):
            gate_type = sim_typelist_3[np.random.randint(0,6)]
            circuit.append(build_gate(gate_type,q))
    Swap | circuit([0, 1])
    Swap | circuit([1, 0])
    Rxx(np.pi) | circuit([2, 3])
    Rxx(np.pi) | circuit([3, 2])
    Ryy(np.pi) | circuit([4, 5])
    Ryy(np.pi) | circuit([5, 4])
    Rzz(np.pi) | circuit([6, 7])
    Rzz(np.pi) | circuit([7, 6])
    Swap | circuit([8, 9])
    Swap | circuit([9, 8])
    CCX | circuit([0, 1, 2])
    CCX | circuit([9, 8, 7])
    CSwap | circuit([3, 4, 5])
    CSwap | circuit([6, 5, 4])

    return circuit
    

cir = build_sim_circuit()
with open('QuICT/wr_unit_test/data.qasm', 'w') as f:
    f.write(cir.qasm())

# amplitude = build_sim_circuit.run(circuit)