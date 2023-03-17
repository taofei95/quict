from QuICT.core import Circuit


qubits = 5     # qubits number
circuit = Circuit(qubits)
circuit.supremacy_append(
    repeat=1,   # The repeat times of cycles
    pattern="ABCDCDAB"  # Indicate the circuit cycle
)

circuit.draw(filename="supremacy")
