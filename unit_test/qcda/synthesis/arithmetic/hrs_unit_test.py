import pytest

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.qcda.synthesis.arithmetic.hrs import (
    HRSAdder,
    HRSAdderMod,
    HRSMulMod,
    CHRSMulMod,
)
from QuICT.simulation.state_vector import ConstantStateVectorSimulator


def set_qureg(qreg_index, N):
    """set qureg to the state |N> in big-endian, same direction as arithmetic circuits

    Args:
        qreg_index (list<int>): _description_
        N (int): _description_

    Returns:
        CompositeGate: a gate converts |0> to |N>
    """
    gate_set = CompositeGate()
    n = len(qreg_index)
    with gate_set:
        for i in range(n):
            if N % 2 == 1:
                X & qreg_index[n - 1 - i]
            N = N // 2
    return gate_set


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
    sim = ConstantStateVectorSimulator()
    for a in range(0, 20):
        for b in range(0, 20):
            n = max(len(bin(a)) - 2, len(bin(b)) - 2)
            circuit = Circuit(n + 2)
            a_q = list(range(n))
            ancilla = n
            ancilla_g = n + 1
            set_qureg(a_q, a) | circuit
            HRSAdder.execute(n, b) | circuit(a_q + [ancilla, ancilla_g])
            Measure | circuit
            sim.run(circuit)
            print("%d + %d = %d\n" % (a, b, int(circuit[a_q])))
            if int(circuit[a_q]) != (a + b) % (2 ** n):
                assert 0
    assert 1


def test_HRSAdderMod():
    sim = ConstantStateVectorSimulator()
    for N in range(4, 15):
        n = len(bin(N)) - 2
        for a in range(0, N):
            for b in range(0, N):
                print(f"{a} + {b} (mod {N}) = ", end="")
                circuit = Circuit(2 * n)
                b_q = list(range(n))
                g_q = list(range(n, 2 * n - 1))
                indicator = 2 * n - 1
                set_qureg(b_q, b) | circuit
                composite_gate = HRSAdderMod.execute(n, a, N)
                composite_gate | circuit(b_q + g_q + [indicator])
                Measure | circuit
                sim.run(circuit)
                print(int(circuit[b_q]))
                if int(circuit[b_q]) != (a + b) % (N):
                    assert 0
    assert 1


def test_HRSMulMod():
    sim = ConstantStateVectorSimulator()
    for N in range(4, 12):
        n = len(bin(N)) - 2
        for a in range(0, N):
            arr = [0, 0]
            if ex_gcd(N, a, arr) != 1:
                continue
            for x in range(0, N):
                print(f"{a} * {x} mod {N} = ", end="")
                circuit = Circuit(2 * n + 1)
                x_q = list(range(n))
                ancilla = list(range(n, 2 * n))
                indicator = 2 * n
                set_qureg(x_q, x) | circuit
                HRSMulMod.execute(n, a, N) | circuit(x_q + ancilla + [indicator])
                Measure | circuit
                sim.run(circuit)
                print(int(circuit[x_q]))
                if int(circuit[x_q]) != (a * x) % (N):
                    print("%d * %d mod %d = %d\n" % (a, x, N, int(x_q)))
                    assert 0
    assert 1


def test_CHRSMulMod():
    sim = ConstantStateVectorSimulator()
    for c in (1, 0):
        for N in range(4, 12):
            n = len(bin(N)) - 2
            for a in range(0, N):
                arr = [0, 0]
                if ex_gcd(N, a, arr) != 1:
                    continue
                for x in range(0, N):
                    print(f"{a} * {x} mod {N} = ", end="")
                    circuit = Circuit(2 * n + 2)
                    x_q = list(range(n))
                    ancilla = list(range(n, 2 * n))
                    indicator = 2 * n
                    control = 2 * n + 1
                    set_qureg(x_q, x) | circuit
                    set_qureg([control], c) | circuit
                    CHRSMulMod.execute(n, a, N) | circuit(
                        x_q + ancilla + [indicator] + [control]
                    )
                    Measure | circuit
                    sim.run(circuit)
                    print(int(circuit[x_q]))
                    if c == 0 and int(circuit[x_q]) != x % (N):
                        print("%d * %d mod %d = %d (c==0)\n" % (a, x, N, int(x_q)))
                        assert 0
                    if c == 1 and int(circuit[x_q]) != (a * x) % (N):
                        print("%d * %d mod %d = %d (c==1)\n" % (a, x, N, int(x_q)))
                        assert 0
    assert 1


if __name__ == "__main__":
    pytest.main(["./unit_test.py"])
