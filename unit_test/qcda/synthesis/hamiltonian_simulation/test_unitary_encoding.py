from QuICT.qcda.synthesis.hamiltonian_simulation.unitary_matrix_encoding import UnitaryMatrixEncoding
import numpy as np
from QuICT.core.gate import X, Y, Z


def test_unitary_matrix_encoding_lcu():
    coefficient_array = np.array([0.3, 0.2])
    matrix_array = np.array([X.matrix, Y.matrix])
    UME = UnitaryMatrixEncoding("LCU")
    circuit = UME.execute(coefficient_array, matrix_array, complete=True, phase_gate=True)
    G, G_inverse, unitary_encoding, unitary_encoding_inverse = UME.execute(coefficient_array, matrix_array,
                                                                           complete=False, phase_gate=True)
    circuit_matrix = circuit.matrix()
    G_matrix = G.matrix()
    unitary_encoding_matrix = unitary_encoding.matrix()
    expected_B_matrix = np.array([[0.77459667 + 0.j, -0.63245553 + 0.j], [0.63245553 + 0.j, 0.77459667 + 0.j]])
    expected_control_v_matrix = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]])
    assert np.allclose(G_matrix[:, 0], np.array([np.sqrt(0.3 / 0.5), np.sqrt(0.2 / 0.5)])), "The G gate isn't correct."
    assert np.allclose(unitary_encoding_matrix, expected_control_v_matrix), "The control-V gate isn't correct"
    expected_B_matrix = np.kron(expected_B_matrix, np.identity(2))
    expected_matrix = np.matmul(np.matmul(np.linalg.inv(expected_B_matrix),
                                          expected_control_v_matrix), expected_B_matrix)
    assert np.allclose(expected_matrix, circuit_matrix), "circuit matrix isn't correct"


def test_unitary_matrix_encoding_conj():
    coefficient_array = np.array([0.3, 0.2])
    matrix_array = np.array([X.matrix, Z.matrix])
    UME = UnitaryMatrixEncoding("conj")
    circuit, circuit_width = UME.execute(coefficient_array, matrix_array)
    circuit_matrix = circuit.matrix()
    circuit_matrix = np.array([[circuit_matrix[0][0], circuit_matrix[0][1]], [circuit_matrix[1][0],
                                circuit_matrix[1][1]]])
    expected_matrix = np.array([[0.2, 0.3], [0.3, -0.2]])
    assert np.allclose(expected_matrix, circuit_matrix), "Circuit matrix doesn't true."
