#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/12 6:33
# @Author  : Han Yu
# @File    : MCT_Linear_Simulation.py

from .._synthesis import Synthesis
from QuICT.core import *

def solve(n, m):
    """

    Args:
        n(int): the number of qubits in the qureg
        m(int): the number of bits of the toffoli

    Returns:
        the circuit which describe the decomposition result
    """
    circuit = Circuit(n)
    if m == 1:
        CX  | circuit([0, n - 1])
    elif m == 2:
        CCX | circuit([0, 1, n - 1])
    else:
        for i in range(m, 2, -1):
            CCX | circuit([i - 1, n - 1 - (m - i + 1), n - 1 - (m - i)])
        CCX | circuit([0, 1, n - m + 1])
        for i in range(3, m + 1):
            CCX | circuit([i - 1, n - 1 - (m - i + 1), n - 1 - (m - i)])

        for i in range(m - 1, 2, -1):
            CCX | circuit([i - 1, n - 1 - (m - i + 1), n - 1 - (m - i)])
        CCX | circuit([0, 1, n - m + 1])
        for i in range(3, m):
            CCX | circuit([i - 1, n - 1 - (m - i + 1), n - 1 - (m - i)])
    return circuit

class MCTLinearSimulationModel(Synthesis):
    """ a linear simulation for toffoli gate

    https://arxiv.org/abs/quant-ph/9503016 Lemma 7.2

    implement m-bit toffoli gate in a qureg with n qubit with linear complexity.

    If n ≥ 5 and m ∈ {3, . . . , ⌈n/2⌉} then (m+1)-Toffoli gate can be simulated
    by a network consisting of 4(m − 2) toffoli gates
    """

    def __call__(self, m):
        """
        Args:
            the bits of the toffoli gate
        Returns:
            MCT_Linear_Simulation_model: model filled by the parameter m.
        """
        self.pargs = [m]
        return self

    def build_gate(self):
        """ overloaded the function "build_gate"

        """
        n = self.targets
        m = self.pargs[0]
        if m > (n // 2) + (1 if n % 2 == 1 else 0):
            raise Exception("control bit cannot above ceil(n/2)")
        if m < 1:
            raise Exception("there must be at least one control bit")
        return solve(self.targets, m)

MCTLinearSimulation = MCTLinearSimulationModel()
