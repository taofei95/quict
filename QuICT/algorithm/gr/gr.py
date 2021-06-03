#!/usr/bin/env python
# -*- coding:utf8 -*-

import numpy as np

from .._algorithm import Algorithm
from QuICT import *
from QuICT.qcda.synthesis.mct import MCTLinearOneDirtyAux

def calcuate_r(nSolution,nSpace):
    '''caculate R

    QCQI p253
    '''
    theta = 2*np.arccos(np.sqrt(1-nSolution/nSpace))
    return round((np.arccos(np.sqrt(nSolution/nSpace))/theta))

def run_standard_grover(f, oracle):
    """ standard grover search with one target

    Args:
        f(list<int>):       the function to be decided
        oracle(function(list<int>,Qureg,Qureg)):
                            the oracle circuit
    Returns:
        int: the target state
    """
    # state space size = (1<<n)
    n = int(np.ceil(np.log2(len(f))))
    circuit = Circuit(n + 1)
    qreg    = circuit([i for i in range(0,n)])
    ancilla = circuit(n)
    # step 1
    H | qreg
    X | ancilla
    H | ancilla
    # step 2: repeat grover iteration R times
    r = calcuate_r(1,1<<n)
    for i in range(r):
        # inversion about target
        oracle(f, qreg, ancilla)
        # inversion about average
        H | qreg
        X | qreg
        H | qreg(n - 1)
        MCTLinearOneDirtyAux.execute(n + 1) | (qreg([j for j in range(0,n - 1)]),qreg(n - 1), ancilla)
        H | qreg(n - 1)
        X | qreg
        H | qreg
    # step 3: Measure
    Measure | qreg
    circuit.exec()
    return int(qreg)

def run_partial_grover():
    pass

class StandardGrover(Algorithm):
    """ standard grover search with one target

    QCQI Chapter 7, p254
    """
    @classmethod
    def run(cls, f, oracle):
        """ standard grover search

        Args:
            f(list<int>):       the function to be decided
            oracle(function):   the oracle
        Returns:
            int: the target state
        """
        return run_standard_grover(f, oracle)

class PartialGrover(Algorithm):
    """ partial grover search with one target

    TODO: ref
    """
    @classmethod
    def run(cls):
        """ partial grover search with one target

        Args:TODO

        Returns:
            int: the target state
        """
        return run_partial_grover()
