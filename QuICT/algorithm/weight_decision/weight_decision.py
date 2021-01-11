import numpy as np

from .._algorithm import Algorithm
from QuICT import *
from QuICT.qcda.synthesis.initial_state_preparation import InitialStatePreparation
from QuICT.qcda.synthesis.mct import MCTOneAux

from QuICT.algorithm import Amplitude

def weight_decison_para(n, k, l):
    kap = k/n
    lam = l/n
    for d in range(n + 2):
        for gamma in range(1, 2*d-2):
            s = (1-np.cos(gamma*np.pi/(2*d-1)))/2
            t = (1-np.cos((gamma+1)*np.pi/(2*d-1)))/2
            if lam*s>=kap*t and (1-kap)*(1-t)>=(1-lam)*(1-s):
                a=np.sqrt((l-k)/(t-s)-(l*s-k*t)/(t-s)-n)
                b=np.sqrt((l*s-k*t)/(t-s))
                return d, gamma, a, b

    print(n, k, l)

def run_weight_decision(f, n, k, l, oracle):
    """ decide function f by k-l algorithm by custom oracle

    https://arxiv.org/abs/1801.05717
    Args:
        f(list<int>): the function to be decided
        n(int): the length of function
        k(int): the smaller weight
        l(int): the bigger weight
        oracle(function): the oracle
    Returns:
        int: the ans, k or l
    """

    num = int(np.ceil(np.log2(n + 2))) + 2
    # Determine number of qreg
    circuit = Circuit(num)
    d, gamma, a, b = weight_decison_para(n, k, l)
    # start the eng and allocate qubits
    qreg = circuit([i for i in range(num - 2)])
    ancilla = circuit(num - 2)
    empty = circuit(num - 1)
    # Start with qreg in equal superposition and ancilla in |->
    N = np.power(2, num - 2)
    value = [0 for _ in range(N)]
    for i in range(n):
        value[i] = 1 / np.sqrt(n + a ** 2 + b ** 2)
    value[N - 2] = a / np.sqrt(n + a ** 2 + b ** 2)
    value[N - 1] = b / np.sqrt(n + a ** 2 + b ** 2)
    print(value)
    # Apply oracle U_f which flips the phase of every state |x> with f(x) = 1
    InitialStatePreparation(value) | qreg
    amp = Amplitude.run(circuit, ancilla=[num - 2, num - 1])
    X | ancilla
    H | ancilla

    print(amp)

    print(abs(np.divide(value, amp)))

    for i in range(d - 1):
        oracle(f, qreg, ancilla)
        MCTOneAux | circuit
        # amp = Amplitude.run(circuit,ancilla=[num-1])
        # print(amp)
        InitialStatePreparation(value) ^ qreg
        X | qreg
        MCTOneAux | circuit
        X | qreg
        InitialStatePreparation(value) | qreg
    # Apply H
    H | ancilla
    X | ancilla
    oracle(f, qreg, ancilla)
    MCTOneAux | circuit
    # Measure

    Measure | qreg
    Measure | ancilla

    circuit.exec()

    y = int(qreg)
    print()
    print(y)
    print(int(ancilla) == gamma % 2)
    if int(ancilla) == gamma % 2:
        return l
    else:
        return k

class WeightDecision(Algorithm):
    @classmethod
    def run(cls, f, n, k, l, oracle):
        """ decide function f by k-l algorithm by custom oracle

        https://arxiv.org/abs/1801.05717
        Args:
            f(list<int>): the function to be decided
            n(int): the length of function
            k(int): the smaller weight
            l(int): the bigger weight
            oracle(function): the oracle
        Returns:
            int: the ans, k or l
        """
        return run_weight_decision(f, n, k, l, oracle)
