from QuICT.core import Circuit
from QuICT.simulation.density_matrix import DensityMatrixSimulator

# Build circuit with 100 random gates and 5 qubits
circuit = Circuit(4)
circuit.random_append(rand_size=50)

# Simulate Quantum Circuit
simulator = DensityMatrixSimulator(
    device="CPU",
    precision="double"
)
result = simulator.run(circuit)
print(result)
sample = simulator.sample(100)
print(sample)
