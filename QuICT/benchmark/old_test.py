
from copy import deepcopy
from itertools import chain
import math
import os

import matplotlib as mpl
from matplotlib import pyplot as plt
import scipy
import scipy.stats
from QuICT.benchmark.benchmark import QuICTBenchmark
from QuICT.benchmark.get_benchmark_circuit import BenchmarkCircuitBuilder
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

bench = QuICTBenchmark("circuit", "Graph-radar")
# # a_list = [GateType.cx, [GateType.h, GateType.rx, GateType.ry, GateType.rz]]
cir_list = bench.get_circuit(["cnf","grover"], 9, 100, 55)
print(len(cir_list))
for k in cir_list:
    print(k.name)
result = [np.load("cnf1.npy"), np.load("cnf2.npy"), np.load("cnf3.npy"), np.load("cnf4.npy"), np.load("grover3.npy"), np.load("grover2.npy"), np.load("grover1.npy")]


ev = bench.evaluate(circuit_list=cir_list, result_list=result)
print(ev)


# cir = BenchmarkCircuitBuilder().mediate_measure_circuit_build(5, 20)
# print(cir)



# def normalization(data):
#     data = np.array(data.get())
#     data = data/np.sum(data)

#     return data



# print(cir_list)
# for cir in cir_list[0]:
#     print(cir.name)

# cir = OPENQASMInterface.load_file("QuICT/lib/circuitlib/algorithm/adder/w7_s8_d5.qasm").circuit
# # print(cir.size())
# sim = ConstantStateVectorSimulator()
# result = sim.run(cir)
# np.save("adder2.npy",result)
# print(result)
# print(normalization(result))

##########
# defaultdict(<class 'list'>, {'algorithm': [[(<QuICT.core.circuit.circuit.Circuit object at 0x7f46a73ecd90>, 
# array([-0.5-6.67640947e-18j,  0. +0.00000000e+00j, -0.5-2.22546982e-18j,
#         0. +0.00000000e+00j, -0.5-2.22546982e-18j,  0. +0.00000000e+00j,
#    -0.5-2.22546982e-18j,  0. +0.00000000e+00j]))]]})
#########

# result = [np.load("grover1.npy"), np.load("grover2.npy")]
# result = [np.load("grover1.npy")]
# print(result[0])
# return sorted(cirs_field_map.items())
# result = [np.load("adder1.npy")]
# ev = bench.evaluate(circuit_list=cir_list[0], result_list=result, output_type="Table")
# print(ev)
# a = ["circuit_width", "circuit_size", "circuit_depth", "qubit_cal", "entropy_cal", "alg_cal", "field_score"]
# for i in ev:
# from prettytable import PrettyTable
# tb = PrettyTable.PrettyTable()
# tb.field_names = ['circuit_width', 'circuit_size', 'circuit_depth', 'qubit_cal', 'entropy_cal', 'alg_cal', 'field_score']
# # tb.field_names = ["circuit_width", "circuit_size", "circuit_depth", "qubit_cal", "entropy_cal", "alg_cal", "field_score"]
# tb.add_row(['1', '2', '3', '4', '5', '6', '7'])
# print(tb)
# import pandas as pd
# data = ['1', '2', '3', '4', '5', '6', '7']
# columns = ['circuit_width', 'circuit_size', 'circuit_depth', 'qubit_cal', 'entropy_cal', 'alg_cal', 'field_score']
# df = pd.DataFrame(data=data,columns=columns)
# print('学生成绩表')
# print(df)

# import prettytable as pt

# # tb = pt.PrettyTable( ["City name", "Area", "Population", "Annual Rainfall"])
# tb = pt.PrettyTable()
# tb.field_names = ['circuit_width', 'circuit_size', 'circuit_depth', 'qubit_cal', 'entropy_cal', 'alg_cal', 'field_score']
# tb.add_row(["Adelaide",1295, 1158259, 600.5, 1, 2, 3])


# print(tb)
    
# for circuit_result_group in  ev:
#     print(circuit_result_group[0])
    # for k in circuit_result_group:
    #     print(k)
# for field, group in ev:
#     print(field, group)


# a = [60, 100, 0]
# b = [0.3, 0.3, 0.4]
# result_list = [x*y for x,y in zip(a, b)]
# print(result_list)
        

# print(ev)

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


# def normalization(data):
#     data = np.array(data)
#     data = data/np.sum(data)

#     return data

# def _kl_cal(p, q):
#         # calculate KL
#         KL_divergence = scipy.stats.entropy(p, q)
#         return KL_divergence
        
# def _cross_en_cal(p, q):
#     # calculate cross E
#     sum=0.0
#     for x in map(lambda y,p:(1-y)*math.log(1-p)+y*math.log(p), p, q):
#         sum+=x
#     cross_entropy = -sum/len(p)
#     return cross_entropy
        
# def _l2_cal(p, q):
#     # calculate L2
#     L2_loss = np.sum(np.square(p - q))
#     return L2_loss


# def circuit_score(circuit, result):
#     # Step 1: simulate circuit
#     simulator=CircuitSimulator()
#     sim_result = normalization(simulator.run(circuit))
    
#     # Step 2: calculate kl, cross_en, l2, qubit
#     mac_result = normalization(result)
#     kl = _kl_cal(np.array(sim_result), np.array(mac_result))
#     cross_en = _cross_en_cal(np.array(sim_result), np.array(mac_result))
#     l2 = _l2_cal(np.array(sim_result), np.array(mac_result))
    
#     # Step 3: return result
#     return kl, cross_en, l2

# for cir in cir_list[0]:
#     for re in result:
#         print(circuit_score(circuit=cir, result=re))

# result_dict = {"circuit_width":1,
#                     "circuit_size": 2,
#                    "circuit_depth": 3
#                 #    "qubit_cal": score_list[0],
#                 #    "entropy_cal": score_list[1],
#                 #    "alg_cal":  score_list[2],
#                 #    "field_score": result_list
#                    }
# for i in result_dict:
#     print(i)
# print(result_dict["circuit_width"])

# a = benchmarkcircuitlib(5, 10, False)
# cir = a.He_circuit_build()
# print(cir.size())

# cir.draw(filename="222")

