import numpy as np
import os
import time
from QuICT.core import Circuit, Layout
from QuICT.core.gate import *
from QuICT.simulation.unitary import *
from QuICT.qcda.optimization import *
from QuICT.tools.interface.qasm_interface import OPENQASMInterface

qubit_num = 5
cir = Circuit(qubit_num)
typelist = [GateType.cx]
cir.random_append(10,typelist=typelist)

# layout = Layout.load_file(
#         os.path.dirname(os.path.abspath(__file__)) + 
#         f"/../qcda-benchmark/layout/line5.layout"
#         )
# cir.topology=layout

#circuit 转 compositecircuit
cir_opt = CnotWithoutAncilla().execute(cir)

cir_q = Circuit(5)
X | cir_q
cir_opt | cir_q
simulator = UnitarySimulator()
b = simulator.run(cir_q)


print(cir.qasm())
print(cir_q.qasm())
print(b)
print(cir.depth())
print(cir_opt.size())


# cir_q = OPENQASMInterface.load_file("wr_unit_test/machine-benchmark/random.qasm").circuit
# print(cir_q.qasm())
# #circuit 转 compositecircuit
# # cir = Circuit(8)
# H | cir_opt(0)
# simulator = UnitarySimulator()
# a = simulator.run(cir_opt)
# print(a)
# print(cir.qasm())

