import numpy as np
import time

from projectq import MainEngine  # import the main compiler engine
from projectq.backends import CircuitDrawer
from projectq.setups.default import get_engine_list
from projectq.ops import H, Measure, CRz, All, QFT, Barrier  # import the operations we want to perform (Hadamard and measurement)


qubits = 20
drawing_engine = CircuitDrawer()
eng = MainEngine(engine_list=get_engine_list() + [drawing_engine])
circ = eng.allocate_qureg(qubits)  # allocate 1 qubit

# for i in range(qubits):
#     H | circ[i]  # apply a Hadamard gate
#     for j in range(i+1, qubits):
#         params = 2 * np.pi / (1 << j - i + 1)
#         CRz(params) | (circ[i], circ[j])

QFT | circ

# Measure | qubit  # measure the qubit
start_time = time.time()
All(Measure) | circ

eng.flush()  # flush all gates (and execute measurements)
end_time = time.time()
print([int(x) for x in circ])  # output measurement result
# print(drawing_engine.get_latex())
print(end_time - start_time)


# from QCompute import *

# # 新建一个量子环境
# env = QEnv() 
# # 量子端可自选，本例使用 CloudBaiduSim2Water
# env.backend(BackendName.LocalBaiduSim2) 

# qubit_num = 20

# q = [env.Q[i] for i in range(qubit_num)]

# # The first step of Grover's search algorithm, superposition
# for i in range(qubit_num):
#     H(q[i])
#     for j in range(i+1, qubit_num):
#         params = 2 * np.pi / (1 << j - i + 1)
#         gate = CRZ(params)
#         gate(q[i], q[j])

# shots = 1

# # Measure with the computational basis;
# # if the user you want to increase the number of Grover iteration,
# # please repeat the code from the comment “Enter the first Grover iteration” to here,
# # and then measure
# MeasureZ(q, range(qubit_num))
# # Commit the quest
# start_time = time.time()
# taskResult = env.commit(shots, fetchMeasure=True)
# end_time = time.time()
# print(taskResult['counts'])
# print(end_time - start_time) 


