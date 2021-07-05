from typing import AbstractSet
import pytest

from QuICT.core import Circuit, X, Measure
from QuICT.qcda.synthesis.arithmetic.bea import *

def Set(qreg, N):
    """
    Set the qreg as N, using X gates on specific qubits
    """
    n = len(qreg)
    for i in range(n):
        if N % 2 == 1:
            X | qreg[n - 1 - i]
        N = N // 2


def ExGCD(a, b, coff):
    if b == 0:
        coff[0] = 1
        coff[1] = 0
        return a
    r = ExGCD(b, a % b, coff)
    t = coff[0]
    coff[0] = coff[1]
    coff[1] = t - a // b * coff[1]
    return r


def test_DraperAdder():
    for a in range(0, 20):
        for b in range(0, 20):
            n = max(len(bin(a)) - 2, len(bin(b)) - 2)
            circuit = Circuit(n * 2)
            qreg_a = circuit([i for i in range(n)])
            qreg_b = circuit([i for i in range(n, n * 2)])
            Set(qreg_a, a)
            Set(qreg_b, b)
            BEAAdder.execute(n) | circuit
            Measure | circuit
            circuit.exec()
            # aa = int(qreg_a)
            bb = int(qreg_b)
            if bb != (a + b)%(2**n):
                print("{0}+{1}={2}".format(str(a), str(b), str(bb)))
                assert 0
    assert 1


def test_FourierAdderWired():
    for a in range(0, 20):
        for b in range(0, 20):
            n = max(len(bin(a)) - 2, len(bin(b)) - 2)
            circuit = Circuit(n + 1)
            qreg_b = circuit([i for i in range(n + 1)])
            Set(qreg_b, b)
            BEAAdderWired.execute(n, a) | circuit
            Measure | circuit
            circuit.exec()
            # aa = int(qreg_a)
            bb = int(qreg_b)
            # print("{0}+{1}={2}".format(str(a), str(b), str(bb)))
            if bb != (a + b)%(2**(n+1)):
                print("{0}+{1}={2}".format(str(a), str(b), str(bb)))
                assert 0
    assert 1


def test_FourierReverseAdderWired():
    for a in range(0, 20):
        for b in range(0, 20):
            n = max(len(bin(a)) - 2, len(bin(b)) - 2)
            circuit = Circuit(n + 1)
            qreg_b = circuit([i for i in range(n + 1)])
            Set(qreg_b, b)
            BEAReverseAdderWired.execute(n, a) | circuit
            Measure | circuit
            circuit.exec()
            # aa = int(qreg_a)
            bb = int(qreg_b)
            if bb != (b-a)%(2**(n+1)):
                print("{0}+{1}={2}".format(str(a), str(b), str(bb)))
                assert 0
    assert 1


def test_FourierAdderMod():
    for N in range(0,20):
        for a in range(0, N):
            for b in range(0, N):
                n = len(bin(N)) - 2
                circuit = Circuit(n + 2)
                qreg_b = circuit([i for i in range(n + 1)])
                Set(qreg_b, b)
                BEAAdderMod.execute(n, a, N) | circuit
                Measure | circuit
                circuit.exec()
                # aa = int(qreg_a)
                bb = int(qreg_b)
                low = int(circuit(n + 1))
                assert low == 0
                # print("({0}+{1}) % {3}={2}".format(str(a), str(b), str(bb),str(N)))
                assert bb == (a + b) % N


def test_BEAMulMod():
    for N in range(0, 20):
        for a in range(0, N):
            for x in range(0, N):
                n = len(bin(N)) - 2
                circuit = Circuit(2 * n + 2)
                qreg_b = circuit([i for i in range(n + 1)])
                qreg_x = circuit([i for i in range(n + 1, 2 * n + 1)])
                Set(qreg_b, 0)
                Set(qreg_x, x)
                BEAMulMod.execute(n, a, N) | circuit
                Measure | circuit
                circuit.exec()
                bb = int(qreg_b)
                # print("0 + {0}*{1} mod {2}={3}".format(str(a), str(x), str(N), str(bb)))
                assert bb == (0 + a * x) % N


def test_BEACUa():
    n = 4
    for c in (1,):
        if c == 0:
            print("disabled")
        else:
            print("enabled")
        for N in range(0, 1<<n):
            for a in range(0, N):
                coff = [0, 0]
                r = ExGCD(a, N, coff)
                if r != 1:
                    continue
                for x in range(0, N):
                    circuit = Circuit(2 * n + 3)
                    qreg_b = circuit([i for i in range(n + 1)])
                    qreg_x = circuit([i for i in range(n + 1, 2 * n + 1)])
                    qreg_c = circuit(2 * n + 1)
                    Set(qreg_c, c)
                    Set(qreg_b, 0)
                    Set(qreg_x, x)
                    BEACUa.execute(n, a, N) | circuit
                    Measure | circuit
                    circuit.exec()
                    xx = int(qreg_x)
                    bb = int(qreg_b)
                    print("{0}*{1} mod {2}={3}".format(str(a), str(x), str(N), str(xx)))
                    if c == 0:
                        assert xx == x
                    else:
                        assert xx == (a * x) % N
                    # assert bb == 0

if __name__ == "__main__":
    pytest.main(["./unit_test.py"])