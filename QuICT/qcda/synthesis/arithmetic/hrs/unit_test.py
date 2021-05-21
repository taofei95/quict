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


def EX_GCD(a, b, arr):
    """ 
    Implementation of Extended Euclidean algorithm

    Args:
        a(int): the parameter a
        b(int): the parameter b
        arr(list): store the solution of ax + by = gcd(a, b) in arr, length is 2

    """

    if b == 0:
        arr[0] = 1
        arr[1] = 0
        return a
    g = EX_GCD(b, a % b, arr)
    t = arr[0]
    arr[0] = arr[1]
    arr[1] = t - int(a / b) * arr[1]
    return g


def test_HRSAdder():
    for a in range(0,20):
        for b in range(0,20):
            n = max(len(bin(a)) - 2, len(bin(b)) - 2)
            circuit = Circuit(n + 2)
            a_q = circuit([i for i in range(n)])
            ancilla = circuit(n)
            ancilla_g = circuit(n + 1)
            Set(a_q, a)
            HRSAdder(n, b) | (a_q, ancilla, ancilla_g)
            Measure | circuit
            circuit.exec()
            if int(a_q) != (a + b)%(2**n):
                print("%d + %d = %d\n"%(a,b,int(a_q)))
                assert 0
    assert 1
            

def test_HRSMulMod():
    arr = [0,0]
    for N in range(0,20):
        for a in range(0,N):
            if EX_GCD(N, a, arr) != 1:
                continue
            for x in range(0,N):
                n = len(bin(N)) - 2
                circuit = Circuit(2*n + 1)
                x_q = circuit([i for i in range(n)])
                ancilla = circuit([i for i in range(n, 2*n)])
                indicator = circuit(2*n)
                Set(x_q, x)
                HRSMulMod(n, a, N) | (x_q, ancilla, indicator)
                Measure | circuit
                circuit.exec()
                if int(x_q) != (a*x)%(N):
                    print("%d * %d mod %d = %d\n"%(a,x,N,int(x_q)))
                    assert 0
    assert 1

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