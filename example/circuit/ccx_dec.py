from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.simulation.state_vector import CircuitSimulator


# Build quantum circuit
circuit = Circuit(3)

X        | circuit(0)
H        | circuit(1)
H        | circuit(2)
CX       | circuit([1, 0])
T_dagger | circuit(0)
CX       | circuit([2, 0])
T        | circuit(0)
CX       | circuit([1, 0])
T_dagger | circuit(0)
CX       | circuit([2, 0])
T        | circuit(0)
T_dagger | circuit(1)
H        | circuit(0)
CX       | circuit([2, 1])
T_dagger | circuit(1)
CX       | circuit([2, 1])
T        | circuit(2)
S        | circuit(1)
H        | circuit(2)
H        | circuit(1)
X        | circuit(2)
X        | circuit(1)
H        | circuit(1)
CX       | circuit([2, 1])
H        | circuit(1)
X        | circuit(2)
X        | circuit(1)
H        | circuit(2)
H        | circuit(1)

# Simulate the quantum circuit by state vector simulator
simulator = CircuitSimulator()
amplitude = simulator.run(circuit=circuit)

print(amplitude)
