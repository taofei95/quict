from QuICT.qcda.synthesis.hamiltonian_simulation import HamiltonianSimulation, trotter
from QuICT.simulation import state_vector
import numpy as np


def test_hamiltonian_simulation():
    z = np.array([[1, 0], [0, 1]])
    y = np.array([[0, -1j], [1j, 0]])
    x = np.array([[0, 1], [1, 0]])
    coefficient_array = np.array([0.2, 0.4])
    unitary_matrix_array = np.array([np.kron(x, y), np.kron(y, z)])
    initial_state = np.array([1, 0, 0, 0])
    HS = HamiltonianSimulation("TS")
    circuit, circuit_dictionary = HS.execute(
        hamiltonian=[coefficient_array, unitary_matrix_array],
        time=23, initial_state=initial_state, error=0.01)
    circuit.draw("command")
    vector = state_vector.StateVectorSimulator()
    vector = vector.run(circuit)
    final_state = np.array([vector[0], vector[1], vector[2], vector[3]])
    assert (np.sum(np.abs(final_state - circuit_dictionary["approximated_time_evolution_operator"][:, 0]))
            < 0.1), "TS method is not correct."

    HS = HamiltonianSimulation("Trotter")
    circuit, _ = HS.execute(
        hamiltonian=[[1, 'X0', 'Y1'], [1, 'Z0', 'X1']],
        time=1,
        initial_state=np.array([0, 0, 0, 1]),
        error=0.05
    )
    final_state = trotter.accurate_final_state(
        [[1, 'X0', 'Y1'], [1, 'Z0', 'X1']],
        1, init_statevec=np.array([0, 0, 0, 1]))
    assert np.allclose(circuit.matrix()[:, 0], final_state), "Trotter method fails"
