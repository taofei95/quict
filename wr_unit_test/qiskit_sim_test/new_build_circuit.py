

import numpy as np
from qiskit import QuantumCircuit
from QuICT.core.gate.gate_builder import build_gate
from quict.QuICT.core.utils.gate_type import GateType
from quict.QuICT.wr_unit_test.build_circuit import build_sim_circuit


def new_build_circuit():
    sim_typelist_1 = [GateType.h,GateType.x,GateType.y,GateType.z,GateType.phase]
    sim_typelist_2 = [GateType.s,GateType.t,GateType.sx,GateType.tdg,GateType.sdg]
    sim_typelist_3 = [GateType.rx,GateType.ry,GateType.rz,GateType.u1,GateType.u2,GateType.u3]
    
    circ = QuantumCircuit(10)
    for i in range(5):
        for q in range(10):
            gate_type = sim_typelist_1[np.random.randint(0,4)]
            circ.append(build_gate(gate_type,q))
    circ.cx(9,8)
    circ.cx(8,9)
    circ.cy(7,6)
    circ.cy(6,7)
    circ.cz(5,4)
    circ.cz(4,5)
    circ.ch(3,2)
    circ.ch(2,3)
    circ.crz(1,0)
    circ.crz(0,1)
    for i in range(5):
        for q in range(10):
            gate_type = sim_typelist_2[np.random.randint(0,5)]
            circ.append(build_gate(gate_type,q))
    circ.rxx(np.pi,9,8)
    circ.rxx(np.pi,8,9)
    circ.ryy(np.pi,7,6)
    circ.ryy(np.pi,6,7)
    circ.rzz(np.pi,5,4)
    circ.rzz(np.pi,4,5)
    circ.cu1(np.pi / 2,3,2)
    circ.cu1(np.pi / 2,2,3)
    circ.cu1(np.pi / 2,0,1,1,0)
    circ.cu1(np.pi / 2,0,1,0,1)
    for i in range(5):
        for q in range(10):
            gate_type = sim_typelist_3[np.random.randint(0,6)]
            circ.append(build_gate(gate_type,q))
    circ.swap(9,8)
    circ.swap(8,9)
    circ.rxx(np.pi,7,6)
    circ.rxx(np.pi,6,7)
    circ.ryy(np.pi,5,4)
    circ.ryy(np.pi,4,5)
    circ.rzz(np.pi,3,2)
    circ.rzz(np.pi,2,3)
    circ.swap(1,0)
    circ.swap(0,1)
    circ.ccx(9, 8, 7)
    circ.ccx(0, 1, 2)
    circ.cswap(6, 5, 4)
    circ.cswap(3, 4, 5)
    circ.draw()
    return circ
    
circ = build_sim_circuit()
with open('QuICT/wr_unit_test/data.qasm', 'w') as f:
    f.write(circ.qasm())


    











    

    