import pytest

from QuICT.algorithm import *
from QuICT.core import *
from QuICT.qcda.synthesis.arithmetic.hrs import *


def Set(qreg, N):
    """
    Set the qreg as N, using X gates on specific qubits
    """
    n = len(qreg)
    for i in range(n):
        if N % 2 == 1:
            X | qreg[n - 1 - i]
        N = N // 2


'''
def test6():
    # n > 2 should obey
    circuit = Circuit(8)
    circuit.assign_initial_zeros()
    X | circuit(0)
    b = 0
    x = 1
    a = 4
    N = 2
    # seems that b = 0 x = 1, the mod computation doesn't work.
    # when b = 0, x = 1, a = 5, N = 4, expected result = 1, but the result we get is 5
    qubit_x = circuit([1, 2, 3])
    qubit_b = circuit([4, 5, 6])
    Set(qubit_x, x)
    Set(qubit_b, b)
    HRSCMulModRaw(3, a, N) | circuit
    Measure | circuit
    circuit.exec()

    qubit_target = circuit([4, 5, 6])
    print(int(qubit_target))
'''


def test7():
    # x * a mod N
    # n = 3
    circuit = Circuit(7)
    circuit.assign_initial_zeros()
    x = 1
    a = 2
    N = 4
    qubit_x = circuit([0, 1, 2])
    Set(qubit_x, x)
    HRSMulMod(3, a, N) | circuit
    Measure | circuit
    circuit.exec()

    qubit_target = circuit([0, 1, 2])
    print(int(qubit_target))


'''
def test1():
    circuit = Circuit(4)
    circuit.assign_initial_zeros()
    X | circuit(1)
    amplitude = Amplitude.run(circuit)
    print(amplitude)
    HRSIncrementer(2) | circuit
    amplitude = Amplitude.run(circuit)
    print(amplitude)
'''


def test2():
    circuit = Circuit(4)
    circuit.assign_initial_zeros()
    HRSAdder(2, 2) | circuit
    amplitude = Amplitude.run(circuit)
    print(amplitude)


'''
def test3():
    circuit = Circuit(5)
    circuit.assign_initial_zeros()
    X | circuit(0) # control = 1 !!!
    HRSCSub(2, 1) | circuit
    amplitude = Amplitude.run(circuit)
    print(amplitude)
'''
'''
def test4():
    circuit = Circuit(6)
    circuit.assign_initial_zeros()
    X | circuit(0) # control1 = 1
    X | circuit(1) # control2 = 1
    # b = 00

    HRSCCCompare(2, 1) | circuit # c = 1

    # expected result: 110001----49

    amplitude = Amplitude.run(circuit)
    print(amplitude)
'''


def test5():
    circuit = Circuit(6)
    circuit.assign_initial_zeros()
    # b = 010 = 2
    X | circuit(3)
    # len(g) needs to be larger than 1, so n needs to be larger than 2 !!!
    # indicator bit somtimes seems to change
    HRSAdderMod(3, 1, 2) | circuit  # a = 1, N = 3, (b + a) % N = 1

    # expected result: 110100----52
    for i in range(6):
        Measure | circuit(i)
    circuit.exec()
    qreg = circuit([i for i in range(6)])
    y = int(qreg)
    print(y)


if __name__ == "__main__":
    pytest.main(["./unit_test.py"])