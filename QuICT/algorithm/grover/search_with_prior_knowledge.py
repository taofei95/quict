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

def run_search_with_prior_knowledge(f, n, p, T, oracle):
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
    global p_global 
    p_global = p[:]
    global T_global
    T_global = T
    num = int(np.ceil(np.log2(n))) + 2
    # Determine number of qreg
    circuit = Circuit(num)
    qreg = circuit([i for i in range(num - 2)])
    ancilla = circuit(num - 2)
    empty = circuit(num - 1)
    cons = ({'type': 'ineq', 'fun': lambda x: 1-np.dot(np.ones(n), x)},)
    bnd = [(0, np.sin(np.pi / (4 * T + 2))**2) for _ in range(n)]
    tmp = np.min([1 / n, np.sin(np.pi/(4 * T + 2))**2])
    x = np.array([tmp] * n)
    option = {'maxiter' : 4, 'disp' : False}
    res = minimize(fun, x, method = 'SLSQP', constraints = cons, bounds = bnd, options = option)
    q = res.x
    
    # Start with qreg in equal superposition and ancilla in |->
    X | ancilla
    H | ancilla
    InitialStatePreparation(list(q)) | qreg
    for i in range(T):
        oracle(f, qreg, ancilla)
        InitialStatePreparation(list(q)) ^ qreg
        X | qreg
        MCTOneAux | circuit
        X | qreg
        InitialStatePreparation(list(q)) | qreg
    # Apply H
    H | ancilla
    X | ancilla
    # Measure
    Measure | qreg
    Measure | ancilla
    Measure | empty
    circuit.exec()
    return int(qreg)

class GroverWithPriorKnowledge(Algorithm):
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
