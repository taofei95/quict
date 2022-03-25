# Author  : Peng Sirui

'''
The (2n+3)-qubit circuit used in the Shor algorithm is designed by \
Stephane Beauregard in "Circuit for Shor's algorithm using 2n+3 qubits"\

Without one control-bit trick.
'''

import logging
from math import pi
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
    n = int(np.ceil(np.log2(N)))
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
        gate_pow = pow(a, 1 << (t - 1 - k), N) # CUa^{2^{t - 1 - k}}
        BEACUa.execute(n, gate_pow, N) | circuit(b_reg+x_reg+[trickbits[k]]) # TODO: break, see if works
    # assert int(circuit[qreg_low]) == 0 and int(circuit[b_reg]) == 0 # TODO: check
    IQFT.build_gate(len(trickbits)) | circuit(trickbits) # TODO: check
    for idx in trickbits:
        Measure | circuit(idx)
    # logging.info(f'\tphi~ (approximately s/r) in binary form is {trickbits_store}')
    return circuit, trickbits


def order_finding(a: int, N: int, eps: float = 1 / 10, simulator: Simulator = CircuitSimulator()):
    """
    The (2n+3)-qubit circuit used in the Shor algorithm is designed by \
    Stephane Beauregard in "Circuit for Shor's algorithm using 2n+3 qubits"

    Quantum algorithm to compute the order of a (mod N), when gcd(a,N)=1.
    """
    circuit, trickbits = construct_circuit(a, N, eps)
    simulator.run(circuit)
    t = len(trickbits)
    trickbits_store = [0] * t
    for idx in trickbits:
        trickbits_store[idx] = int(circuit[trickbits[idx]]) # TODO: check
    trickbits_store.reverse()
 
    # continued fraction procedure
    phi_ = sum([(trickbits_store[i] * 1. / (1 << (i + 1))) for i in range(t)])
    logging.info(f'\tphi~ (approximately s/r) in decimal form is {phi_}')
    r = Fraction(phi_).limit_denominator(N - 1).denominator
    logging.info(f'\tclose fraction form: {Fraction(phi_).limit_denominator(N - 1)}')
    return r
