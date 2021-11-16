import pytest

from QuICT.core import Circuit, X, Measure
from QuICT.qcda.synthesis.arithmetic.hrs import HRSAdder, HRSAdderMod, HRSMulMod


def set_qureg(qreg, N):
    """
    Set the qreg as N, using X gates on specific qubits
    """
    n = len(qreg)
    for i in range(n):
        if N % 2 == 1:
            X | qreg[n - 1 - i]
        N = N // 2


def ex_gcd(a, b, arr):
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
    g = ex_gcd(b, a % b, arr)
    t = arr[0]
    arr[0] = arr[1]
    arr[1] = t - int(a / b) * arr[1]
    return g

def test_HRSAdder():
    for a in range(0, 20):
        for b in range(0, 20):
            n = max(len(bin(a)) - 2, len(bin(b)) - 2)
            circuit = Circuit(n + 2)
            a_q = circuit(list(range(n)))
            ancilla = circuit(n)
            ancilla_g = circuit(n + 1)
            set_qureg(a_q, a)
            HRSAdder.execute(n, b) | (a_q, ancilla, ancilla_g)
            Measure | circuit
            circuit.exec()
            if int(a_q) != (a + b) % (2 ** n):
                print("%d + %d = %d\n" % (a, b, int(a_q)))
                assert 0
    assert 1


def test_HRSAdderMod():
    for N in range(4, 15):
        n = len(bin(N)) - 2
        for a in range(0, N):
            for b in range(0, N):
                print("%d + %d (mod %d)= " % (a, b, N))
                circuit = Circuit(2 * n)
                b_q = circuit(list(range(n)))
                g_q = circuit(list(range(n, 2 * n - 1)))
                indicator = circuit(2 * n - 1)
                set_qureg(b_q, b)
                composite_gate = HRSAdderMod.execute(n, a, N) 
                composite_gate | (b_q, g_q, indicator)
                Measure | circuit
                circuit.exec()
                print(int(b_q))
                if int(b_q) != (a + b) % (N):
                    assert 0
    assert 1

def test_HRSMulMod():
    arr = [0, 0]
    for N in range(4, 12):
        n = len(bin(N)) - 2
        for a in range(0, N):
            if ex_gcd(N, a, arr) != 1:
                continue
            for x in range(0, N):
                print("%d * %d mod %d = " % (a, x, N))
                circuit = Circuit(2 * n + 1)
                x_q = circuit(list(range(n)))
                ancilla = circuit(list(range(n, 2 * n)))
                indicator = circuit(2 * n)
                set_qureg(x_q, x)
                HRSMulMod.execute(n, a, N) | (x_q, ancilla, indicator)
                Measure | circuit
                circuit.exec()
                if int(x_q) != (a * x) % (N):
                    print("%d * %d mod %d = %d\n" % (a, x, N, int(x_q)))
                    assert 0
    assert 1



if __name__ == "__main__":
    pytest.main(["./unit_test.py"])
    