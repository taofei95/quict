# Author  : Peng Sirui

'''
The (2n+3)-qubit circuit used in the Shor algorithm is designed by \
St´ephane Beauregard in "Circuit for Shor’s algorithm using 2n+3 qubits"\
'''

import logging
from math import pi
import numpy as np
from fractions import Fraction

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.qcda.synthesis.arithmetic.bea import *
from QuICT.algorithm import Algorithm
from .utility import *

from QuICT.simulation.cpu_simulator import CircuitSimulator
from QuICT.simulation import Simulator

def order_finding(a: int, N: int, eps: float = 1 / 10, simulator: Simulator = CircuitSimulator()):
    """
    The (2n+3)-qubit circuit used in the Shor algorithm is designed by \
    Stephane Beauregard in "Circuit for Shor's algorithm using 2n+3 qubits"

    Quantum algorithm to compute the order of a (mod N), when gcd(a,N)=1.
    """
    # phase estimation procedure
    n = int(np.ceil(np.log2(N)))
    t = int(2 * n + 1 + np.ceil(np.log(2 + 1 / (2 * eps))))
    logging.info(f'\torder_finding begin: circuit: n = {n} t = {t}')
    trickbit_store = [0] * t

    circuit = Circuit(2 * n + 3)
    b_reg = [i for i in range(n + 1)]
    x_reg = [i for i in range(n + 1, 2 * n + 1)]
    trickbit = [2 * n + 1]
    qreg_low = [2 * n + 2]
    X | circuit(x_reg[n - 1])
    for k in range(t):
        H | circuit(trickbit)
        gate_pow = pow(a, 1 << (t - 1 - k), N)
        BEACUa.execute(n, gate_pow, N) | circuit
        for i in range(k):
            if trickbit_store[i]:
                Rz(-pi / (1 << (k - i))) | circuit(trickbit)
        H | circuit(trickbit)

        for idx in (b_reg + trickbit + qreg_low): Measure | circuit(idx)
        simulator.run(circuit)
        assert int(circuit[qreg_low]) == 0 and int(circuit[b_reg]) == 0
        logging.info(f'\tthe {k}th trickbit measured to be {int(circuit[trickbit])}')
        trickbit_store[k] = int(circuit[trickbit])
        if trickbit_store[k] == 1:
            X | circuit(trickbit)
    for idx in x_reg: Measure | circuit(idx)

    trickbit_store.reverse()
    logging.info(f'\tphi~ (approximately s/r) in binary form is {trickbit_store}')
    
    # continued fraction procedure
    phi_ = sum([(trickbit_store[i] * 1. / (1 << (i + 1))) for i in range(t)])
    logging.info(f'\tphi~ (approximately s/r) in decimal form is {phi_}')
    if phi_ == 0.0:
        logging.info('\torder_finding failed: phi~ = 0')
        return 0
    (num, den) = (Fraction(phi_).numerator, Fraction(phi_).denominator)
    CFE = continued_fraction_expansion(num, den) #TODO: using Fraction.limit_denominator
    logging.info(f'\tContinued fraction expansion of phi~ is {CFE}')
    num1, den1, num2, den2 = CFE[0], 1, 1, 0
    logging.info(f'\tthe 0th convergence is {num1}/{den1}')
    for k in range(1, len(CFE)):
        num = num1 * CFE[k] + num2
        den = den1 * CFE[k] + den2
        logging.info(f'\tthe {k}th convergence is {num}/{den}')
        if den >= N:
            break
        else:
            num2 = num1
            num1 = num
            den2 = den1
            den1 = den
    r = den1
    return r


class BEA_order_finding_twice(Algorithm):
    '''
    Run order_finding twice and take the lcm of the two result
    to guaruntee a higher possibility to get the correct order,
    as suggested in QCQI 5.3.1
    '''
    @staticmethod
    def run(a: int, N: int, demo: str = None, eps: float = 1 / 10, simulator: Simulator = CircuitSimulator()):
        r1 = order_finding(a, N, demo, eps, simulator)
        r2 = order_finding(a, N, demo, eps, simulator)
        flag1 = (pow(a, r1, N) == 1 and r1 != 0)
        flag2 = (pow(a, r2, N) == 1 and r2 != 0)
        if flag1 and flag2:
            r = min(r1, r2)
        elif not flag1 and not flag2:
            r = int(np.lcm(r1, r2))
        else:
            r = int(flag1) * r1 + int(flag2) * r2

        if (pow(a, r, N) == 1 and r != 0):
            msg = f'\torder_finding found candidate order: r = {r} of a = {a}'
        else:
            r = 0
            msg = '\torder_finding failed'
        logging.info(msg)
        return r
