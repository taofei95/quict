# Author  : Zhu Qinlin

'''
The (2n+2)-qubit circuit used in the Shor algorithm is designed \
by THOMAS HANER, MARTIN ROETTELER, and KRYSTA M. SVORE in \
"Factoring using 2n+2 qubits with Toffoli based modular multiplication"
'''

import logging
from fractions import Fraction
from math import pi
import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.qcda.synthesis.arithmetic.hrs import *
from QuICT.simulation import Simulator
from QuICT.simulation.cpu_simulator import CircuitSimulator
from QuICT.core.operator import Trigger, CheckPoint, CheckPointChild

from .utility import *


def construct_circuit(a: int, N: int, eps: float = 1 / 10):
    # phase estimation procedure
    n = int(np.ceil(np.log2(N + 1)))
    t = int(2 * n + 1 + np.ceil(np.log(2 + 1 / (2 * eps))))
    logging.info(f'\tcircuit construction begin: circuit: n = {n} t = {t}')

    circuit = Circuit(2 * n + 2)
    x_reg = list(range(n))
    ancilla = list(range(n,2*n))
    indicator = 2*n
    trickbit = [2 * n + 1]

    # cgates[measured qubit index][added gate index][measured result]
    cgates = [
        [[CompositeGate() for measured in range(2)] for j in range(t+1)] for i in range(t)
    ]
    # triggers[measured qubit index][added gate index]
    triggers = [[
            Trigger(
                1, [cgates[i][j][measured] for measured in range(2)],True
            ) for j in range(t+1)
        ] for i in range(t)
    ]
    triggers_reset = list(range(t))
    # checkpoints[measured qubit index]
    checkpoints = [CheckPoint() for i in range(t)]
    checkpoints_reset = [CheckPoint() for i in range(t)]

    # triggers!!!
    for k in range(t): # end at k-th checkpoints
        for i in range(k): # start at i-th qubits (compressed)
            cpc = checkpoints[k].get_child()
            Rz(-np.pi / (1 << (k - i))) | cgates[i][k-1-i][1](trickbit)
            cpc | cgates[i][k-1-i][1]
    # triggers for reset!!!
    for i in range(t):
        cgate = CompositeGate()
        X | cgate(trickbit)
        triggers_reset[i] = Trigger(1, [CompositeGate(), cgate], True)

    # subcircuit: init \ket{1}\ket{0}
    X | circuit(x_reg[n - 1])

    for k in range(t):
        # subcircuit CUa
        H | circuit(trickbit)
        gate_pow = pow(a, 1 << (t - 1 - k), N)
        CHRSMulMod.execute(n, gate_pow, N) | circuit
        # subcircuit: semi-classical QFT
        checkpoints[k] | circuit
        H | circuit(trickbit)
        # subcircuit: measure & reset trickbit
        for i in range(t-k):
            triggers[k][i] | circuit(trickbit)
        triggers_reset[k] | circuit(trickbit)
    return circuit, [triggers[i][0] for i in range(t)][::-1]

def order_finding(a: int, N: int, eps: float = 1 / 10, simulator: Simulator = CircuitSimulator()):
    """
    Shor algorithm by THOMAS HANER, MARTIN ROETTELER, and KRYSTA M. SVORE \
    in "Factoring using 2n+2 qubits with Toffoli based modular multiplication"
    Quantum algorithm to compute the order of a (mod N), when gcd(a,N)=1.
    """
    # phase estimation procedure
    n = int(np.ceil(np.log2(N + 1)))
    t = int(2 * n + 1 + np.ceil(np.log(2 + 1 / (2 * eps))))
    logging.info(f'\torder_finding begin: circuit: n = {n} t = {t}')
    trickbit_store = [0] * t

    circuit = Circuit(2 * n + 2)
    x_reg = list(range(n))
    ancilla = list(range(n,2*n))
    indicator = 2*n
    trickbit = [2 * n + 1]
    # subcircuit: init \ket{1}\ket{0}
    circuit = Circuit(2 * n + 2)
    X | circuit(x_reg[n - 1])
    amp = simulator.run(circuit)

    for k in range(t):
        # subcircuit CUa
        circuit = Circuit(2 * n + 2)
        H | circuit(trickbit)
        gate_pow = pow(a, 1 << (t - 1 - k), N)
        CHRSMulMod.execute(n, gate_pow, N) | circuit
        amp = simulator.run(circuit, True)
        # subcircuit: semi-classical QFT
        circuit = Circuit(2 * n + 2)
        for i in range(k):
            if trickbit_store[i]:
                Rz(-pi / (1 << (k - i))) | circuit(trickbit)
        H | circuit(trickbit)
        amp = simulator.run(circuit, True)
        circuit = Circuit(2 * n + 2)
        for idx in [trickbit]+ancilla+[indicator]:
            Measure | circuit(idx)
        amp = simulator.run(circuit, True)
        # subcircuit: measure & reset trickbit
        assert int(circuit[indicator]) == 0 and int(circuit[ancilla]) == 0
        logging.info(f'\tthe {k}th trickbit measured to be {int(circuit[trickbit])}')
        trickbit_store[k] = int(circuit[trickbit])
        if trickbit_store[k] == 1:
            circuit = Circuit(2 * n + 2)
            X | circuit(trickbit)
            simulator.run(circuit, True)

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
