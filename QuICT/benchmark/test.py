
from copy import deepcopy
from itertools import chain
import os

import matplotlib as mpl
from matplotlib import pyplot as plt
from QuICT.benchmark.benchmark import QuICTBenchmark
from QuICT.core.gate.gate import *
from QuICT.core.utils.gate_type import GateType
from QuICT.simulation.simulator import Simulator
from QuICT.simulation.state_vector.cpu_simulator.cpu import CircuitSimulator
from QuICT.simulation.state_vector.gpu_simulator.constant_statevector_simulator import ConstantStateVectorSimulator
from QuICT.simulation.unitary.unitary_simulator import UnitarySimulator
from QuICT.tools.circuit_library import *
from QuICT.qcda.qcda import QCDA
from QuICT.qcda.synthesis.gate_transform.instruction_set import InstructionSet
from QuICT.tools.interface.qasm_interface import OPENQASMInterface

bench = QuICTBenchmark("circuit", "Graph")
a_list = [GateType.cx, [GateType.h, GateType.rx, GateType.ry, GateType.rz]]
cir_list = bench.get_circuit(["google"], 3, 20, 10, InSet=a_list)
result = [[-0.25-0.10355339j, 0.60355339-0.25j, 0.25+0.10355339j, -0.60355339+0.25j], 
[-2.50000000e-01-0.25j, -1.38777878e-16+0.85355339j, -1.11022302e-16-0.14644661j -2.50000000e-01+0.25j]]
# print(cir_list)
# for i in cir_list[1]:
#     print(i.name)

ev = bench.evaluate(circuit_list=cir_list[0], result_list=result)
print(ev)

# for circuit_result_group in  ev:
#     # for circuit, result in circuit_result_group:
#     print(circuit_result_group[0].qasm())
                
# for k, v in ev:
#     print(k, v)

# [{'instructionset': [<QuICT.core.circuit.circuit.Circuit object at 0x7f1d3c5b9d00>, [(-0.25-0.25j), (-1.38777878e-16+0.85355339j), (-0.2500000000000001+0.10355339j)]]}, {'instructionset': [<QuICT.core.circuit.circuit.Circuit object at 0x7f1d56ea7a60>, [(-0.25-0.25j), (-1.38777878e-16+0.85355339j), (-0.2500000000000001+0.10355339j)]]}]    

# for cir in cir_list:
#     cir_list2, cir_list_onequbit, cir_list_twoqubit, InSet , cir_list3, one_list = [], [], [], [], [], []
#     qcda = QCDA()
# #     # InSet_list = [CX, H]
# #     # for gate in InSet_list:
# #     #     if gate.targets + gate.controls == 2:
# #     #         cir_list_twoqubit.append(gate.type)
# #     #     else:
# #     #         cir_list_onequbit.append(gate.type)
# #     # for two in cir_list_onequbit:
# #     #     for one in cir_list_onequbit:
# #     #         one_list.append(one)
# a_list = [GateType.cx, [GateType.h, GateType.rx, GateType.ry, GateType.rz]]
#     q_ins = InstructionSet(a_list[0], a_list[1])

#     qcda.add_gate_transform(q_ins)
    
#     cir_list2 = qcda.compile(cir)
#     cir_list3.append(cir_list2)
        
# print(cir_list)



# circuit_type = "template"
# a = CircuitLib(circuit_type)
# print(a)
# cir_list = CircuitLib().get_algorithm_circuit("grover", 5, 20, 20)
# print(cir_list)

# filePath = 'QuICT/lib/circuitlib/circuit_qasm/algorithm'
# a = os.listdir(filePath)
# c = []
# for b in a:
#     if b == '.keep':
#         continue
#     c.append(b)
# print(c)

# a_list = {"a":2, "b":3}
# for a in a_list:
#     print(a)

# file_path = 'QuICT/lib/circuitlib/circuit_qasm'
# alg_file_list = os.listdir(file_path)

# print(alg_file_list)
# for value in CircuitLib().__DEFAULT_CLASSIFY:
#     print(value)

# a = QuICTBenchmark().show_result()

# print(CircuitLib().size)

# cir_list = CircuitLib("circuit").get_benchmark_circuit("highly_parallelized", 5, 20, 20)
# for cir in cir_list:
#     print(cir.name)

# list1=[1,2,3]
# list2=[4]
# list3=[5]
# list_all = list(chain(list1,list2,list3))
# print(list_all)

# cdb = CircuitLibDB()
# # cdb.clean()
# for t in ['random', 'algorithm', 'benchmark', 'instructionset']:
#     cdb.add_circuit(t)

# cdb.add_template_circuit()
# print(CircuitLib().size)

# cir = OPENQASMInterface.load_file("unit_test/simulation/data/random_circuit_for_correction.qasm").circuit
# sim = ConstantStateVectorSimulator()
# U = sim.run(deepcopy(cir))
# print(U)


# u_sim = Simulator(device="CPU", precision="double", backend="unitary")
# u = u_sim.run(cir)
# print(u["data"]["state_vector"])


