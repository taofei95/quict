#from itertools import permutations
import numpy as np
from QuICT.core.gate import DiagonalGate

#from QuICT.core.gate import *
from QuICT.core import Circuit

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
    circuit = Circuit(n)
    gates | circuit
    circuit.draw(filename='test_phase_shift_no_aux_1.jpg',flatten=True)

def test_phase_shift_with_aux():
    n = 3
    theta = 2 * np.pi * np.random.random(1 << n)
    seq = np.random.permutation(np.arange(1, 1 << n))
    gates = DiagonalGate.phase_shift(theta, seq, aux=n)
    assert np.allclose(theta, np.mod(np.angle(np.diagonal(gates.matrix()))[::2], 2 * np.pi))
    circuit = Circuit(n*2)
    gates | circuit
    circuit.draw(filename='test_phase_shift_with_aux_2.jpg', flatten=True)

def test_partitioned_gray_code():
    n = 4
    m = 8
    t = int(np.floor(np.log2(m / 2)))
    s = DiagonalGate.partitioned_gray_code(n,t)
    print(s)
    #print(s[1][0]) #s(2,1)

def test_alpha_s():
    n = 4
    s = 6
    theta = 2 * np.pi * np.random.random(1 << n)
    #theta = 2 * np.pi * np.random.random(len(A_inv))

    print(DiagonalGate.alpha_s(theta, s, n))

def test_phase_shift_s():
    n = 4
    s = 6
    theta = 2 * np.pi * np.random.random(1 << n)
    alpha = DiagonalGate.alpha_s(theta, s, n)
    gates = DiagonalGate.phase_shift_s(s, n, alpha,aux=n)
    circuit = Circuit(n*2)
    gates | circuit
    circuit.draw(filename='test_phase_shift_s_3.jpg', flatten=True)

def test_linear_fjk():
    n = 4
    m = 8
    t = int(np.floor(np.log2(m / 2)))
    resu = DiagonalGate.linear_fjk(2,1,10,n,t)
    print(resu)

if __name__ == '__main__':
    #test_gray_code()
    #test_Ainv()
    #test_phase_shift_no_aux() #need change the size of A_inv
    #test_phase_shift_with_aux() #need change the size of A_inv
    #test_partitioned_gray_code()
    #test_linear_fjk()
    #test_alpha_s()
    test_phase_shift_s() #here dim(A_inv)=dim(theta)

