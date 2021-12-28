from QuICT.core import *
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

a = int(input('input a: '))
x = int(input('input b: '))
N = int(input('input b: '))

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