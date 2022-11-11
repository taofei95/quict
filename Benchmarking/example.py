import os
from QuICT.Benchmarking.Benchmarking import Benchmarking
from QuICT.core.circuit.circuit import Circuit
from QuICT.core.gate.gate import *
from QuICT.core.layout.layout import Layout
from QuICT.core.utils.gate_type import GateType
from QuICT.qcda.qcda import QCDA
from QuICT.qcda.synthesis.gate_transform.instruction_set import InstructionSet
from QuICT.tools.interface.qasm_interface import OPENQASMInterface
from QuICT.simulation.state_vector import ConstantStateVectorSimulator


"""QuICT Benchmarking example"""
#指令集
# q_ins = InstructionSet(
#             GateType.cx,
#             [GateType.h, GateType.rx, GateType.ry, GateType.rz]
#         )
        
# cir = OPENQASMInterface.load_file("QuICT/Benchmarking/circuitlib/special_circuits/mediate_measure_circuit.qasm").circuit
# # InSet = GateType.cx, [GateType.h, GateType.rx, GateType.ry, GateType.rz]
# # cir_run = Benchmarking().run(cir, bench_synthesis=True)
# f = open('Mm_circuit_amplitude.txt','w+')
# sim = ConstantStateVectorSimulator()
# sim_data = sim.run(cir).get()
# f.write(str(sim_data))
# f.close()

gate_multiply = []
for i in range(5, 11):
        gate_multiply.append(i)
for j in range(11, 50):
        if j%2==0:
                gate_multiply.append(j)
for m in range(50, 201):
        if m % 10 ==0:
                gate_multiply.append(m)


path = "QuICT/Benchmarking/circuitlib/random_circuits/ibmq_set/"
dirs = os.listdir( path )
for file in dirs:
    cir = OPENQASMInterface.load_file("QuICT/Benchmarking/circuitlib/random_circuits/ibmq_set/" + file).circuit

for q_num in range(2, 3):
    for gm in gate_multiply:
        dep = cir.depth()
        folder_path = "QuICT/Benchmarking/amplib/random_circuits/ibmq_set/"
        f = open(folder_path + '/' + f'q{q_num}-g{gm*q_num}-d{dep}-amplitude.txt','w+')
        sim = ConstantStateVectorSimulator()
        sim_data = sim.run(cir).get()
        f.write(str(sim_data))
        f.close()
