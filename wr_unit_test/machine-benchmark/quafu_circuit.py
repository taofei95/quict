import os
from QuICT.core.circuit.circuit import Circuit
from QuICT.qcda.optimization import *
from QuICT.simulation.unitary import *
from QuICT.core.utils.gate_type import GateType


single_typelist = [
    GateType.x, GateType.y, GateType.z, GateType.h,
    GateType.t, GateType.tdg, GateType.s, GateType.sdg,
    GateType.sx, GateType.rx, GateType.ry, GateType.rz
]
double_typelist = [GateType.cx, GateType.cz, GateType.swap]
len_s, len_d = len(single_typelist), len(double_typelist)
prob = [0.8 / len_s] * len_s + [0.2 / len_d] * len_d

qubit_num = 5
cir = Circuit(qubit_num)
cir.random_append(rand_size=20, typelist=single_typelist+double_typelist, probabilities=prob)

cir_opt = CommutativeOptimization().execute(cir)

sim = UnitarySimulator()
amp = sim.run(cir_opt)
print(amp)

print(cir.qasm())
print(cir_opt.qasm())
print(cir_opt.size())
print(cir_opt.depth())
print(f"cir depth:{cir.depth()}")