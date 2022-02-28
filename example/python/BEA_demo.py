# modified from HRS_demo.py by Zhu Qinlin
from QuICT.core import Circuit, X, Measure
from QuICT.qcda.synthesis.arithmetic.bea import *


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


def BEAAdder_demonstration():
    print('BEAAdder demonstration: a(quantum) + b(classical)')
    a = int(input('\tinput quantum number a: '))
    b = int(input('\tinput classical number b: '))
    n = int(input('\tinput quantum number length n: '))

    circuit = Circuit(n + 1)
    qreg_b = circuit(list(range(n + 1)))
    set_qureg(qreg_b, a)
    BEAAdderWired.execute(n, b) | circuit
    Measure | qreg_b

    circuit.draw('matp', 'BEAAdder_circuit.jpg')
    circuit.exec()

    print("\t%d + %d = %d" % (a, b, int(qreg_b)))


def BEAAdderMod_demonstration():
    print('BEAAdderMod demonstration: a(quantum) + b(classical) mod N(classical)')
    a = int(input('\tinput quantum number a: '))
    b = int(input('\tinput classical number b: '))
    N = int(input('\tinput classical modulo N: '))

    if a >= N or b >= N:
        print('\ta >= N or b >= N not allowed')
        return 0

    n = len(bin(N)) - 2
    circuit = Circuit(n + 2)
    qreg_b = circuit(list(range(n + 1)))
    set_qureg(qreg_b, a)
    BEAAdderMod.execute(n, b, N) | circuit
    Measure | circuit
    circuit.exec()
    bb = int(qreg_b)
    low = int(circuit(n + 1))

    print("\t%d + %d (mod %d) = %d" % (a, b, N, bb))


def BEAMulMod_demonstration():
    print('BEAMulMod demonstration: For gcd(a,N) = 1, a(classical)*x(quantum) mod N(classical)')
    a = int(input('\tinput classical number a: '))
    x = int(input('\tinput quantum number x: '))
    N = int(input('\tinput classical modulo N: '))

    arr = [0, 0]
    if ex_gcd(N, a, arr) != 1:
        print('\tgcd(a,N) != 1')
        return 0

    n = len(bin(N)) - 2
    circuit = Circuit(2 * n + 2)
    qreg_b = circuit(list(range(n + 1)))
    qreg_x = circuit(list(range(n + 1, 2 * n + 1)))
    set_qureg(qreg_b, 0)
    set_qureg(qreg_x, x)
    BEAMulMod.execute(n, a, N) | circuit
    Measure | circuit

    # circuit.draw('matp','BEAMulMod_circuit.jpg') #the image too large
    circuit.exec()

    print("\t%d * %d (mod %d) = %d" % (a, x, N, int(qreg_b)))


BEAAdder_demonstration()
BEAAdderMod_demonstration()
BEAMulMod_demonstration()
