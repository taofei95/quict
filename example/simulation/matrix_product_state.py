from QuICT.core import Circuit
from QuICT.simulation.matrix_product_state import MatrixProductStateSimulator


# Build circuit with 100 random gates and 5 qubits
circuit = Circuit(4)
circuit.random_append(rand_size=50)

# Simulate Quantum Circuit
simulator = MatrixProductStateSimulator("CPU")
result = simulator.run(circuit)
print(result)
sample = simulator.sample(100)
print(sample)
