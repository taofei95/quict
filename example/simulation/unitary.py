from QuICT.core import Circuit
from QuICT.simulation.unitary import UnitarySimulator

# Build circuit with 100 random gates and 5 qubits
circuit = Circuit(4)
circuit.random_append(rand_size=100)

# Simulate Quantum Circuit
simulator = UnitarySimulator(
    device="CPU",
    precision="double"
)
result = simulator.run(circuit)
print(result)
sample = simulator.sample(1000)
print(sample)
