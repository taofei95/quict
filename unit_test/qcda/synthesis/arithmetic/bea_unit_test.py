import pytest

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.qcda.synthesis.arithmetic.bea import (
    BEAAdder,
    BEAAdderWired,
    BEAReverseAdderWired,
    BEAAdderMod,
    BEAMulMod,
    BEACUa
)
from QuICT.simulation.state_vector import ConstantStateVectorSimulator


def set_qureg(qreg_index, N):
    """using X to set qureg to N.

    Args:
        qreg_index (list): _description_
        N (int): _description_

    Returns:
        CompositeGate: _description_
    """
    gate_set = CompositeGate()
    n = len(qreg_index)
    with gate_set:
        for i in range(n):
            if N % 2 == 1:
                X & qreg_index[n - 1 - i]
            N = N // 2
    return gate_set


def ex_gcd(a, b, coff):
    if b == 0:
        coff[0] = 1
        coff[1] = 0
        return a
    r = ex_gcd(b, a % b, coff)
    t = coff[0]
    coff[0] = coff[1]
    coff[1] = t - a // b * coff[1]
    return r


def test_DraperAdder():
    for a in range(0, 20):
        for b in range(0, 20):
            n = max(len(bin(a)) - 2, len(bin(b)) - 2)
            circuit = Circuit(n * 2)
            qreg_a = list(range(n))
            qreg_b = list(range(n, n * 2))
            set_qureg(qreg_a, a) | circuit
            set_qureg(qreg_b, b) | circuit
            BEAAdder.execute(n) | circuit
            Measure | circuit
            ConstantStateVectorSimulator().run(circuit)
            # aa = int(qreg_a)
            bb = int(circuit[qreg_b])
            if bb != (a + b) % (2 ** n):
                print("{0}+{1}={2}".format(str(a), str(b), str(bb)))
                assert 0
    assert 1


def test_FourierAdderWired():
    for a in range(0, 20):
        for b in range(0, 20):
            n = max(len(bin(a)) - 2, len(bin(b)) - 2)
            circuit = Circuit(n + 1)
            qreg_b = list(range(n + 1))
            set_qureg(qreg_b, b) | circuit
            BEAAdderWired.execute(n, a) | circuit
            Measure | circuit
            ConstantStateVectorSimulator().run(circuit)
            # aa = int(qreg_a)
            bb = int(circuit[qreg_b])
            # print("{0}+{1}={2}".format(str(a), str(b), str(bb)))
            if bb != (a + b) % (2 ** (n + 1)):
                print("{0}+{1}={2}".format(str(a), str(b), str(bb)))
                assert 0
    assert 1


def test_FourierReverseAdderWired():
    for a in range(0, 20):
        for b in range(0, 20):
            n = max(len(bin(a)) - 2, len(bin(b)) - 2)
            circuit = Circuit(n + 1)
            qreg_b = list(range(n + 1))
            set_qureg(qreg_b, b) | circuit
            BEAReverseAdderWired.execute(n, a) | circuit
            Measure | circuit
            ConstantStateVectorSimulator().run(circuit)
            # aa = int(qreg_a)
            bb = int(circuit[qreg_b])
            if bb != (b - a) % (2 ** (n + 1)):
                print("{0}+{1}={2}".format(str(a), str(b), str(bb)))
                assert 0
    assert 1


def test_FourierAdderMod():
    for N in range(0, 20):
        for a in range(0, N):
            for b in range(0, N):
                n = len(bin(N)) - 2
                circuit = Circuit(n + 2)
                qreg_b = list(range(n + 1))
                set_qureg(qreg_b, b) | circuit
                BEAAdderMod.execute(n, a, N) | circuit
                Measure | circuit
                ConstantStateVectorSimulator().run(circuit)
                # aa = int(qreg_a)
                bb = int(circuit[qreg_b])
                low = int(circuit[n + 1])
                assert low == 0
                # print("({0}+{1}) % {3}={2}".format(str(a), str(b), str(bb),str(N)))
                assert bb == (a + b) % N


def test_BEAMulMod():
    for N in range(0, 20):
        for a in range(0, N):
            for x in range(0, N):
                n = len(bin(N)) - 2
                circuit = Circuit(2 * n + 2)
                qreg_b = list(range(n + 1))
                qreg_x = list(range(n + 1, 2 * n + 1))
                set_qureg(qreg_b, 0) | circuit
                set_qureg(qreg_x, x) | circuit
                BEAMulMod.execute(n, a, N) | circuit
                Measure | circuit
                ConstantStateVectorSimulator().run(circuit)
                bb = int(circuit[qreg_b])
                # print("0 + {0}*{1} mod {2}={3}".format(str(a), str(x), str(N), str(bb)))
                assert bb == (0 + a * x) % N


def test_BEACUa():
    n = 4
    for c in range(2):
        if c == 0:
            print("disabled")
        else:
            print("enabled")
        for N in range(0, 1 << n):
            for a in range(0, N):
                coff = [0, 0]
                r = ex_gcd(a, N, coff)
                if r != 1:
                    continue
                for x in range(0, N):
                    print("%d^%d * %d mod %d" % (a, c, x, N))
                    circuit = Circuit(2 * n + 3)
                    qreg_b = list(range(n + 1))
                    qreg_x = list(range(n + 1, 2 * n + 1))
                    qreg_c = [2 * n + 1]
                    qreg_low = [2 * n + 2]
                    set_qureg(qreg_c, c) | circuit
                    set_qureg(qreg_b, 0) | circuit
                    set_qureg(qreg_x, x) | circuit
                    BEACUa.execute(n, a, N) | circuit
                    Measure | circuit
                    ConstantStateVectorSimulator().run(circuit)
                    bb = int(circuit[qreg_b])
                    xx = int(circuit[qreg_x])
                    cc = int(circuit[qreg_c])
                    low = int(circuit[qreg_low])
                    print("b = {0}, x = {1} , c = {2}, low = {3}".format(str(bb), str(xx), str(cc), str(low)))
                    if c == 0:
                        assert (bb == 0) and (xx == x) and (cc == c) and (low == 0)
                    else:
                        assert (bb == 0) and (xx == (a * x) % N) and (cc == c) and (low == 0)


if __name__ == "__main__":
    pytest.main(["./unit_test.py"])
