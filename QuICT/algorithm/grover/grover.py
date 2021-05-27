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
from QuICT.qcda.synthesis.mct import MCTOneAux
    
p_global = []
T_global = 1

def fun(x):
    return -np.dot(p_global, np.sin((2*T_global+1)*np.arcsin(np.sqrt(x)))**2)

def run_grover(f, n, p, T, oracle):
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
    round = 1

    H | circuit
    for i in range(round):
        #Grover iteration
        oracle(index_q,result_q)
        H | index_q
        #control phase shift
        X | index_q
        MCTOneAux.execute(n+1) | circuit

class PartialGrover(Algorithm):
    """ grover search with prior knowledge

    https://arxiv.org/abs/2009.08721

    """
    @classmethod
    def run(cls, f, n, p, T, oracle):
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
        return run_search_with_prior_knowledge(f, n, p, T, oracle)
