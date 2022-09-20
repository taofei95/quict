import time
from QuICT import core
from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.simulation.state_vector.gpu_simulator.constant_statevector_simulator import ConstantStateVectorSimulator



qubit_num = 25
gate_size = 800
circuit = Circuit(qubit_num)
circuit.random_append(gate_size)

# circuit = Circuit(5)
# H | circuit
# CX | circuit([0, 1])
# CX | circuit([1, 0])
# CY | circuit([2, 3])
# CY | circuit([3, 2])
# CZ | circuit([4, 3])


# simulator = ConstantStateVectorSimulator(precision="single")
simulator_double = ConstantStateVectorSimulator(precision="double")

# sv = simulator_double.run(circuit).get()
# print(sv)


# stime = time.time()
# simulator.run(circuit)
# etime = time.time()

stime1 = time.time()
simulator_double.run(circuit)
etime1 = time.time()

# print(f"QuICT time single:{etime - stime}")
print(f"QuICT time double:{etime1 - stime1}")
