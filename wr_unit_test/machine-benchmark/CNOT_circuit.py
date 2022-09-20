import numpy as np
import os
import time
from QuICT.core import Circuit, Layout
from QuICT.core.gate import *
from QuICT.simulation.unitary import *
from QuICT.qcda.optimization import *
from QuICT.tools.interface.qasm_interface import OPENQASMInterface

# qubit_num = 5
# cir = Circuit(qubit_num)
# typelist = [GateType.cx]
# cir.random_append(10,typelist=typelist)
# CX | cir([0, 1])
# CX | cir([1, 2])
# CX | cir([2, 3])
# CX | cir([3, 4])
# CX | cir([4, 5])
# CX | cir([5, 6])
# CX | cir([6, 7])

# # Rz(np.pi / 4) | cir([5, 6])
# # CX | cir([4, 5])
# # CX | cir([3, 4])
# CX | cir([2, 3])
# CX | cir([1, 2])
# CX | cir([0, 1])

# qubit_num = 7
# cir = Circuit(qubit_num)
# typelist = [GateType.cx]
# cir.random_append(15,typelist=typelist)
# print(cir.qasm())

# layout = Layout.load_file(
#         os.path.dirname(os.path.abspath(__file__)) + 
#         f"/../qcda-benchmark/layout/line7.layout"
#         )
# cir.topology=layout
# cir_opt = TopologicalCnot().execute(cir)
# print(cir_opt.qasm())


cir_q = OPENQASMInterface.load_file("wr_unit_test/machine-benchmark/random.qasm").circuit
print(cir_q.qasm())
#circuit 转 compositecircuit
# cir = Circuit(8)
# H | cir(0)
# cir_opt | cir
simulator = UnitarySimulator()
a = simulator.run(cir_q)
print(a)





# stime = time.time()
# cir.draw(filename='CNOT circuit')
# print(cir.qasm())
# Amplitude = CircuitSimulator().run(cir)
# print(Amplitude)
# stime = time.time()
# simulator = UnitarySimulator()
# a = simulator.run(cir)
# ltime = time.time()
# print(ltime - stime)
# print(a)

# cir_opt = CnotWithoutAncilla().execute(cir)
# cir_opt.draw(filename = 'cnot')
# simulator = UnitarySimulator()
# b = simulator.run(cir_opt)
# ltime = time.time()
# print(ltime - stime)
# print(b)