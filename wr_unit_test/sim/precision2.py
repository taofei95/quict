

from QuICT.core.gate import *
from QuICT.core import Circuit
from QuICT.simulation.state_vector.gpu_simulator.constant_statevector_simulator import ConstantStateVectorSimulator

qubit_num=3
circuit = Circuit(qubit_num)
H | circuit
CX | circuit([1, 0])
CX | circuit([0, 1])
CH | circuit([1, 0])
Swap | circuit([2, 1])
SX | circuit(0)
T | circuit(1)
T_dagger | circuit(0)
X | circuit(1)
Y | circuit(1)
S | circuit(2)
U1(np.pi / 2) | circuit(2)
U3(np.pi, 0, 1) | circuit(0)
Rx(np.pi) | circuit(1)
Ry(np.pi / 2) | circuit(2)
Rz(np.pi / 4) | circuit(0)

simulator = ConstantStateVectorSimulator(precision="double")
sv = simulator.run(circuit).get()
a = simulator.sample(100)
print(a)