#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/2/23 10:14 上午
# @Author  : Han Yu
# @File    : Deutsch_Jozsa.py

import numpy as np
from scipy.optimize import minimize

from .._algorithm import Algorithm
from QuICT import *
from QuICT.qcda.synthesis.initial_state_preparation import InitialStatePreparation
from QuICT.qcda.synthesis.mct import MCTLinearOneDirtyAux
    
p_global = []
T_global = 1

def fun(x):
    return -np.dot(p_global, np.sin((2*T_global+1)*np.arcsin(np.sqrt(x)))**2)

def run_grover(f, n, oracle):
    """ grover search for f with custom oracle

    Args:
        f(list<int>): the function to be decided
        n(int): the length of input
        p(list<float>): the distribute of priop probability
        T(int): the query times
        oracle(function): the oracle
    Returns:
        int: the a satisfies that f(a) = 1
    """
    circuit = Circuit(n + 1)
    index_q = circuit([i for i in range(n)])
    result_q = circuit(n)
    N = 2**n
    theta = 2*np.arccos(np.sqrt(1-1/N))
    T = round(np.arccos(np.sqrt(1/N))/theta)

    # create equal superposition state in index_q
    H | index_q 
    #print("equal superposition state made: ", Amplitude.run(circuit))
    # create |-> in result_q
    X | result_q
    H | result_q
    #print("before grover iteration state: ", Amplitude.run(circuit))
    for i in range(T):
        #Grover iteration
        oracle(f, index_q, result_q)
        H | index_q
        #control phase shift
        X | index_q
        H | index_q(n - 1)
        MCTLinearOneDirtyAux.execute(n + 1) | (index_q([j for j in range(0,n - 1)]),index_q(n - 1), result_q)
        H | index_q(n - 1)
        X | index_q
        #control phase shift end
        H | index_q
        #print("After %dth GI, the state is: " %i)
        #print(Amplitude.run(circuit))
    Measure | index_q
    circuit.exec()
    #amplitute = Amplitude.run(circuit)
    #print(amplitute)
    return int(index_q)

class Grover(Algorithm):
    """ grover search with prior knowledge

    https://arxiv.org/abs/2009.08721

    """
    @classmethod
    def run(cls, f, n, oracle):
        """ grover search for f with custom oracle

        Args:
            f(list<int>): the function to be decided
            n(int): the length of input
            p(list<float>): the distribute of priop probability
            T(int): the query times
            oracle(function): the oracle
        Returns:
            int: the a satisfies that f(a) = 1
        """
        return run_grover(f, n, oracle)
