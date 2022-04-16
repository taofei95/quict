# Author  : Peng Sirui

'''
The (2n+3)-qubit circuit used in the Shor algorithm is designed by \
St´ephane Beauregard in "Circuit for Shor’s algorithm using 2n+3 qubits"\
'''

import logging
from math import pi, gcd
import numpy as np
from fractions import Fraction
from functools import reduce

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.qcda.synthesis.arithmetic.bea import *
from QuICT.algorithm import Algorithm
from .utility import *

from QuICT.simulation.cpu_simulator import CircuitSimulator
from QuICT.simulation import Simulator

MAX_ROUND = 3

def reinforced_order_finding(a: int, N: int, eps: float = 1 / 10, simulator: Simulator = CircuitSimulator()):
    r_list = []
    i = 0
    while i < MAX_ROUND:
        i += 1
        r = order_finding(a,N,eps,simulator)
        if r!=0 and (a**r)%N==1:
            logging.info(f'\tsuccess!')
            r_list.append(r)
    if len(r_list) == 0:
        return 0
    else:
        return reduce(lambda x,y: (x*y)//gcd(x,y),r_list)

def order_finding(a: int, N: int, eps: float = 1 / 10, simulator: Simulator = CircuitSimulator()):
    """
    The (2n+3)-qubit circuit used in the Shor algorithm is designed by \
    Stephane Beauregard in "Circuit for Shor's algorithm using 2n+3 qubits"

    Quantum algorithm to compute the order of a (mod N), when gcd(a,N)=1.
    """
    # phase estimation procedure
    n = int(np.ceil(np.log2(N + 1)))
    t = int(2 * n + 1 + np.ceil(np.log(2 + 1 / (2 * eps))))
    logging.info(f'\torder_finding begin: circuit: n = {n} t = {t}')
    trickbit_store = [0] * t

    b_reg = [i for i in range(n + 1)]
    x_reg = [i for i in range(n + 1, 2 * n + 1)]
    trickbit = [2 * n + 1]
    qreg_low = [2 * n + 2]
    # subcircuit: init \ket{1}\ket{0}
    circuit = Circuit(2 * n + 3)
    X | circuit(x_reg[n - 1])
    amp = simulator.run(circuit)

    for k in range(t):
        # subcircuit CUa
        circuit = Circuit(2 * n + 3)
        H | circuit(trickbit)
        gate_pow = pow(a, 1 << (t - 1 - k), N)
        BEACUa.execute(n, gate_pow, N) | circuit
        amp = simulator.run(circuit, use_previous=True)
        # subcircuit: semi-classical QFT
        circuit = Circuit(2 * n + 3)
        for i in range(k):
            if trickbit_store[i]:
                Rz(-pi / (1 << (k - i))) | circuit(trickbit) #TODO: check if it should be append this way
        H | circuit(trickbit)
        amp = simulator.run(circuit, use_previous=True)
        circuit = Circuit(2 * n + 3)
        for idx in (b_reg + trickbit + qreg_low): Measure | circuit(idx)
        amp = simulator.run(circuit, use_previous=True)
        # subcircuit: measure & reset trickbit
        assert int(circuit[qreg_low]) == 0 and int(circuit[b_reg]) == 0
        logging.info(f'\tthe {k}th trickbit measured to be {int(circuit[trickbit])}')
        trickbit_store[k] = int(circuit[trickbit])
        if trickbit_store[k] == 1:
            circuit = Circuit(2 * n + 3)
            X | circuit(trickbit)
            simulator.run(circuit, use_previous=True)

    # for idx in x_reg: Measure | circuit(idx)
    trickbit_store.reverse()
    logging.info(f'\tphi~ (approximately s/r) in binary form is {trickbit_store}')
    # continued fraction procedure
    phi_ = sum([(trickbit_store[i] * 1. / (1 << (i + 1))) for i in range(t)])
    logging.info(f'\tphi~ (approximately s/r) in decimal form is {phi_}')
    if phi_ == 0.0:
        logging.info('\torder_finding failed: phi~ = 0')
        return 0
    frac = Fraction(phi_).limit_denominator(N)
    logging.info(f'\tContinued fraction expansion of phi~ is {frac}')
    return frac.denominator


class BEA_order_finding_twice(Algorithm):
    '''
    Run order_finding twice and take the lcm of the two result
    to guaruntee a higher possibility to get the correct order,
    as suggested in QCQI 5.3.1
    '''
    @staticmethod
    def run(a: int, N: int, demo: str = None, eps: float = 1 / 10, simulator: Simulator = CircuitSimulator()):
        r1 = order_finding(a, N, eps, simulator)
        r2 = order_finding(a, N, eps, simulator)
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
