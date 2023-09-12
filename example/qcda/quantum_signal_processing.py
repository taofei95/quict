from QuICT.qcda.synthesis.quantum_signal_processing import *
from numpy.polynomial.polynomial import Polynomial
import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    SAF = SignalAngleFinder()

    x = np.linspace(-1,1,100)
    P = Polynomial([0,1/8, 0, 0, 0, 1/4, 0,1/16, 0, 1/16, 0, 1/2])
    angle_sequence_length = len(P)-1
    angle_sequence = SAF.execute(polynomial_p= P , k = angle_sequence_length)
    print(angle_sequence)
    QSP = QuantumSignalProcessing(angle_sequence)
    y = []
    for element in x:
        y.append(QSP.signal_matrix(element)[0][0])
    plt.plot(x, P(x))
    plt.plot(x, np.array(y))
    plt.plot(x, np.array(y)-P(x))#compute difference between signal and expectation
    plt.show()

    P = Polynomial([0, 1 / 3, 0, 1/3, 0, 1 / 3])
    angle_sequence_length = len(P) - 1
    angle_sequence = SAF.execute(polynomial_p=P, k=angle_sequence_length)
    print(angle_sequence)
    QSP = QuantumSignalProcessing(angle_sequence)
    y = []
    for element in x:
        y.append(QSP.signal_matrix(element)[0][0])
    plt.plot(x, P(x))
    plt.plot(x, np.array(y))
    plt.plot(x, np.array(y)-P(x))#compute difference between signal and expectation
    plt.show()

    P = Polynomial([0, -1 / 3, 0, -1/3, 0, -1 / 3])
    angle_sequence_length = len(P) - 1
    angle_sequence = SAF.execute(polynomial_p=P, k=angle_sequence_length)
    print(angle_sequence)
    QSP = QuantumSignalProcessing(angle_sequence)
    y = []
    for element in x:
        print(QSP.signal_matrix(element)[0][0])
        y.append(QSP.signal_matrix(element)[0][0])
    plt.plot(x, P(x))
    plt.plot(x, np.array(y))
    plt.plot(x, P(x)-np.array(y))
    plt.show()

    #demo on convert x to P(x)
    coefficient_array = np.array([0.7, 0.2])
    matrix_array = np.array([np.identity(2).astype("complex128"), np.array([[0,1],[1,0]]).astype('complex128')])

    circuit = QSP.signal_processing_circuit(coefficient_array, matrix_array)
    circuit.draw("command")
    print(circuit.matrix()[:,0])
    print(np.matmul(circuit.matrix(), np.array([1] + [0 for i in range(1, 2 ** 3)])).reshape(2 ** 3, 1))