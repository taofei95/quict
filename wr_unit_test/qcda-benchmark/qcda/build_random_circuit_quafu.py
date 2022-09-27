from QuICT.core.circuit.circuit import Circuit
from QuICT.qcda.optimization import *
from QuICT.simulation.unitary import *
from QuICT.core.utils.gate_type import GateType


quafu_typelist = [
    GateType.x, GateType.y, GateType.z, GateType.h,
    GateType.t, GateType.tdg, GateType.s, GateType.sdg,
    GateType.sx, GateType.cx, GateType.cz,
    GateType.swap, GateType.rx, GateType.ry, GateType.rz
]
qubit_num = 5
cir = Circuit(qubit_num)
cir.random_append(rand_size=15, typelist=quafu_typelist)
cir.draw(filename="quafu_5")
# print(cir.qasm())

cir_opt = CommutativeOptimization().execute(cir)
# print(cir_opt.qasm())

sim = UnitarySimulator()
amp = sim.run(cir_opt)
print(amp)
print(cir.qasm())
print(cir_opt.qasm())
print(cir.size(), cir_opt.size())
print(cir.depth(), cir_opt.depth())