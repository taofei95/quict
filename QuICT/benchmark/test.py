
from itertools import chain
import os

import matplotlib as mpl
from matplotlib import pyplot as plt
from QuICT.benchmark import Benchmarking
from QuICT.benchmark.benchmark import QuICTBenchmark
from QuICT.core.gate.gate import *
from QuICT.core.utils.gate_type import GateType
from QuICT.lib.circuitlib.circuit_lib_sql import CircuitLibDB
from QuICT.lib.circuitlib.circuitlib import CircuitLib
from QuICT.qcda.qcda import QCDA
from QuICT.qcda.synthesis.gate_transform.instruction_set import InstructionSet

# bench = QuICTBenchmark("circuit", "Graph")
# # a_list = [GateType.cx, [GateType.h, GateType.rx, GateType.ry, GateType.rz]]
# cir_list, _ = bench.get_circuit(["highly_entangled"], 10, 20, 20)

# ev = bench.evaluate(cir_list)
# print(ev)
    
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文

x_axis_data = [1, 2, 3, 4, 5]
y_axis_data = [1, 2, 3, 4, 5]

# plot中参数的含义分别是横轴值，纵轴值，线的形状，颜色，透明度,线的宽度和标签
plt.plot(x_axis_data, y_axis_data, 'ro-', color='#4169E1', alpha=0.8, linewidth=1, label='一些数字')

# 显示标签，如果不加这句，即使在plot中加了label='一些数字'的参数，最终还是不会显示标签
plt.legend(loc="upper right")
plt.xlabel('x轴数字')
plt.ylabel('y轴数字')

plt.savefig('benchmark line show.jpg') 
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
