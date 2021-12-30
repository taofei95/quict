from QuICT.core import Circuit, X, Measure
from QuICT.qcda.synthesis.arithmetic.tmvh import *
from QuICT.qcda.synthesis.arithmetic.hrs import *

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


def HRSAdder_demonstration():
    print('HRSAdder demonstration: a(quantum) + b(classical)')
    a = int(input('\tinput quantum number a: '))
    b = int(input('\tinput classical number b: '))
    n = int(input('\tinput quantum number length n: '))
    
    circuit = Circuit(n + 2)
    a_q = circuit(list(range(n)))
    ancilla = circuit(n)
    ancilla_g = circuit(n + 1)
    set_qureg(a_q, a)
    HRSAdder.execute(n, b) | (a_q, ancilla, ancilla_g)
    Measure | a_q
    
    circuit.draw('matp','HRSAdder_circuit.jpg')
    circuit.exec()

    print("\t%d + %d = %d" % (a, b, int(a_q)))


def HRSAdderMod_demonstration():
    print('HRSAdderMod demonstration: a(quantum) + b(classical) mod N(classical)')
    a = int(input('\tinput quantum number a: '))
    b = int(input('\tinput classical number b: '))
    N = int(input('\tinput classical modulo N: '))

    n = len(bin(N)) - 2
    circuit = Circuit(2 * n)
    a_q = circuit(list(range(n)))
    g_q = circuit(list(range(n, 2 * n - 1)))
    indicator = circuit(2 * n - 1)
    set_qureg(a_q, a)
    composite_gate = HRSAdderMod.execute(n, b, N)
    composite_gate | (a_q, g_q, indicator)
    Measure | a_q
    
    circuit.draw('matp','HRSAdderMod_circuit.jpg')
    circuit.exec()

    print("\t%d + %d (mod %d) = %d" % (a, b, N, int(a_q)))


def HRSMulMod_demonstration():
    print('HRSMulMod demonstration: For gcd(a,N) = 1, a(classical)*x(quantum) mod N(classical)')
    a = int(input('\tinput classical number a: '))
    x = int(input('\tinput quantum number x: '))
    N = int(input('\tinput classical modulo N: '))

    arr = [0, 0]
    if ex_gcd(N, a, arr) != 1:
        print('\tgcd(a,N) != 1')
        return 0
    
    n = len(bin(N)) - 2
    circuit = Circuit(2 * n + 1)
    x_q = circuit(list(range(n)))
    ancilla = circuit(list(range(n, 2 * n)))
    indicator = circuit(2 * n)
    set_qureg(x_q, x)
    HRSMulMod.execute(n, a, N) | (x_q, ancilla, indicator)
    Measure | circuit
    
    #circuit.draw('matp','HRSMulMod_circuit.jpg') #the image too large
    circuit.exec()

    print("\t%d * %d (mod %d) = %d" % (a, x, N, int(x_q)))


'''
a = int(input('\tinput a: '))
x = int(input('\tinput b: '))
N = int(input('\tinput N: '))

n = len(bin(N)) - 2
circuit = Circuit(2 * n + 1)
x_q = circuit(list(range(n)))
ancilla = circuit(list(range(n, 2 * n)))
indicator = circuit(2 * n)
set_qureg(x_q, x)
HRSMulMod.execute(n, a, N) | (x_q, ancilla, indicator)
Measure | circuit
#circuit.draw()
circuit.exec()

print(int(x_q))
'''
#HRSAdder_demonstration()
#HRSAdderMod_demonstration()
HRSMulMod_demonstration()