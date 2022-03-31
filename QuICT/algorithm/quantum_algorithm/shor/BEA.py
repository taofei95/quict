# Author  : Peng Sirui

'''
The (2n+3)-qubit circuit used in the Shor algorithm is designed by \
Stephane Beauregard in "Circuit for Shor's algorithm using 2n+3 qubits"\

Without one control-bit trick.
'''

import logging
from math import gcd, pi
import numpy as np
from fractions import Fraction

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.qcda.synthesis.arithmetic.bea import *
from .utility import *

from QuICT.simulation.cpu_simulator import CircuitSimulator
from QuICT.simulation import Simulator


def construct_circuit(a: int, N: int, eps: float = 1 / 10):
    # phase estimation procedure
    n = int(np.ceil(np.log2(N + 1)))
    t = int(2 * n + 1 + np.ceil(np.log(2 + 1 / (2 * eps))))
    logging.info(f'\torder_finding circuit construction: circuit: n = {n} t = {t}')

    circuit = Circuit(2 * n + 2 + t)
    b_reg = list(range(n + 1)) # n+1
    x_reg = list(range(n + 1, 2 * n + 1)) # n
    trickbits = list(range(2 * n + 1, 2 * n + 1 + t)) # t
    qreg_low = [2 * n + 1 + t] # 1
    X | circuit(x_reg[n - 1])

    for idx in trickbits:
        H | circuit(idx)
    for k in range(t):
        gate_pow = pow(a, 1 << (t-1-k), N) # CUa^{2^{t-1-k}}
        BEACUa.execute(n, gate_pow, N) | circuit(b_reg+x_reg+[trickbits[k]]+qreg_low)
    IQFT.build_gate(len(trickbits)) | circuit(trickbits)
    for idx in (b_reg + trickbits + qreg_low):
        Measure | circuit(idx)
    return circuit, trickbits[::-1] # for int(circuit[trickbits]) convenience


def order_finding(a: int, N: int, eps: float = 1 / 10, simulator: Simulator = CircuitSimulator()):
    circuit, trickbits = construct_circuit(a, N, eps)
    amp = simulator.run(circuit)
    t = len(trickbits)

    # continued fraction procedure
    phi_ =  int(circuit[trickbits]) / (1 << t)
    logging.info(f'\tphi~ (approximately s/r) in decimal form is {phi_}')
    r = Fraction(phi_).limit_denominator(N - 1).denominator
    logging.info(f'\tclose fraction form: {Fraction(phi_).limit_denominator(N - 1)}')
    return r

MAX_ROUND = 3

def reinforced_order_finding(a: int, N: int, eps: float = 1 / 10, simulator: Simulator = CircuitSimulator()):
    circuit, trickbits = construct_circuit(a, N, eps)
    t = len(trickbits)
    # continued fraction procedure (repetition)
    r_list = []
    i = 0
    while i < MAX_ROUND:
        i += 1
        amp = simulator.run(circuit)
        phi_ =  int(circuit[trickbits]) / (1 << t)
        logging.info(f'\tclose fraction form (repetition {i}): {Fraction(phi_).limit_denominator(N - 1)}')
        r = Fraction(phi_).limit_denominator(N - 1).denominator
        if r!=0 and (a**r)%N==1:
            logging.info(f'\tsuccess!')
            r_list.append(r)
        if len(r_list)>1:
            break
    if len(r_list) == 0:
        return 0
    return min(r_list)
