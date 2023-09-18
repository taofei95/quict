from QuICT.qcda.synthesis.hamiltonian_simulation import *
import numpy as np
from QuICT.simulation import state_vector

if __name__ == '__main__':
    # H = 1 * (X_0 pauli tensor Y_1) + 1 * (Z_0 pauli tensor X_1)
    HS = HamiltonianSimulation("Trotter")
    circuit, _ = HS.execute(hamiltonian=[[1, 'X0', 'Y1'], [1, 'Z0', 'X1']],
                            time=1, initial_state=[0, 0, 0, 1], error=0.05)
    circuit.draw("command")
    print(circuit.matrix())

    x = np.array([[1, 0], [0, 1]])
    y = np.array([[0, -1j], [1j, 0]])
    z = np.array([[0, 1], [1, 0]])
    coefficient_array = np.array([0.2, 0.3, 0.1])
    unitary_matrix_array = np.array([x, y, z])
    initial_state = np.array([0, 1])
    HS = HamiltonianSimulation("TS")
    circuit, circuit_dictionary = HS.execute(hamiltonian=[coefficient_array, unitary_matrix_array],
                                            time=50,
                                            initial_state=initial_state,
                                            error=0.1)
    circuit.draw("command")
    vector = state_vector.StateVectorSimulator()
    vector = vector.run(circuit)
    # The vectors map the state in the order[000,001,010,011...]
    # We need to measure ancilla qubit in 00 state. so in this example, we only keep the
    # first two coefficient in the vector.
    final_state = np.array([vector[0], vector[1]])
    print("calculated state:", final_state)
    print("Expect state:", np.matmul(circuit_dictionary["approximated_time_evolution_operator"], initial_state))
