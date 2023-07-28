from itertools import permutations

import numpy as np

from QuICT.core.gate import DiagonalGate


def test_gray_code():
    for code in DiagonalGate.lucal_gray_code(0, 3):
        print(code)
    print(DiagonalGate.partitioned_gray_code(4, 2))


def test_Ainv():
    n = 3
    A = np.zeros((1 << n, 1 << n))
    for s in range(1, 1 << n):
        for x in range(1, 1 << n):
            A[s, x] = DiagonalGate.binary_inner_prod(s, x, width=n)
    A = A[1:, 1:]
    # A_inv = 2^(1-n) (2A - J)
    A_inv = (2 * A - 1) / (1 << (n - 1))
    print(A)
    print(A_inv)
    print(np.dot(A, A_inv))


def test_phase_shift_no_aux():
    n = 3
    theta = 2 * np.pi * np.random.random(1 << n)
    seq = np.random.permutation(np.arange(1, 1 << n))
    gates = DiagonalGate.phase_shift(theta, seq)
    assert np.allclose(theta, np.mod(np.angle(np.diagonal(gates.matrix())), 2 * np.pi))


def test_phase_shift_with_aux():
    n = 3
    theta = 2 * np.pi * np.random.random(1 << n)
    seq = np.random.permutation(np.arange(1, 1 << n))
    gates = DiagonalGate.phase_shift(theta, seq, aux=n)
    assert np.allclose(theta, np.mod(np.angle(np.diagonal(gates.matrix()))[::2], 2 * np.pi))


if __name__ == '__main__':
    test_gray_code()
    # test_Ainv()
    # test_phase_shift_no_aux()
    # test_phase_shift_with_aux()
