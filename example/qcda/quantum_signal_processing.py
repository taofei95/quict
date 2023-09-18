from QuICT.qcda.synthesis.quantum_signal_processing.quantum_signal_processing import *
from numpy.polynomial.polynomial import Polynomial
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    SAF = SignalAngleFinder()

    x = np.linspace(-1, 1, 100)
    P = Polynomial([0, 1 / 8, 0, 0, 0, 1 / 4, 0, 1 / 16, 0, 1 / 16, 0, 1 / 2])
    angle_sequence_length = len(P) - 1
    angle_sequence = SAF.execute(polynomial_p=P, k=angle_sequence_length)
    print(angle_sequence)
    QSP = QuantumSignalProcessing(angle_sequence)
    y = []
    for element in x:
        y.append(QSP.signal_matrix(element)[0][0])
    plt.plot(x, P(x))
    plt.plot(x, np.array(y))
    # compute difference between signal and expectation
    plt.plot(x, np.array(y) - P(x))
    plt.show()

    P = Polynomial([0, 1 / 3, 0, 1 / 3, 0, 1 / 3])
    angle_sequence_length = len(P) - 1
    angle_sequence = SAF.execute(polynomial_p=P, k=angle_sequence_length)
    print(angle_sequence)
    QSP = QuantumSignalProcessing(angle_sequence)
    y = []
    for element in x:
        y.append(QSP.signal_matrix(element)[0][0])
    plt.plot(x, P(x))
    plt.plot(x, np.array(y))
    # compute difference between signal and expectation
    plt.plot(x, np.array(y) - P(x))
    plt.show()

    P = Polynomial([0, -1 / 3, 0, -1 / 3, 0, -1 / 3])
    angle_sequence_length = len(P) - 1
    angle_sequence = SAF.execute(polynomial_p=P, k=angle_sequence_length)
    print(angle_sequence)
    QSP = QuantumSignalProcessing(angle_sequence)
    y = []
    for element in x:
        y.append(QSP.signal_matrix(element)[0][0])
    plt.plot(x, P(x))
    plt.plot(x, np.array(y))
    plt.plot(x, P(x) - np.array(y))
    plt.show()

    # demo on convert x to P(x)
    QSP = QuantumSignalProcessing(convert_phase_sequence(angle_sequence))
    coefficient_array = np.array([0.4, 0.3])
    matrix_array = np.array([np.identity(2).astype(
        "complex128"), np.array([[0, 1], [1, 0]]).astype('complex128')])

    circuit = QSP.signal_processing_circuit(coefficient_array, matrix_array)
    circuit.draw("command")
    circuit_matrix = circuit.matrix()

    rescaled_matrix = coefficient_array[0] * \
        matrix_array[0] + coefficient_array[1] * matrix_array[1]
    rescaled_matrix = rescaled_matrix / np.sum(coefficient_array)
    eigenvalue, eigenvector = np.linalg.eigh(rescaled_matrix)
    new_matrix = []
    for i in range(len(eigenvalue)):
        eigenvalue[i] = P(eigenvalue[i])
        new_matrix.append(np.kron(eigenvector[i].conj().reshape(
            len(eigenvector), 1), eigenvector[i]) * eigenvalue[i])
    new_matrix = np.sum(np.array(new_matrix), axis=0)

    print(new_matrix[:, 0])
    print(circuit_matrix[:, 0])
    final_state = np.array([circuit_matrix[:, 0][0], circuit_matrix[:, 0][1]])
    print("The deviation between two vectors are:", np.abs(
        np.sum(np.abs(final_state - new_matrix[:, 0]))))
