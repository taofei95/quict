#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/12 6:33
# @Author  : Han Yu
# @File    : MCT_Linear_Simulation.py

from .._synthesis import Synthesis
from QuICT.core import *

def HalfDirtyAux(n, m, controls, auxs, target):
    """

    Args:
        n(int): the number of qubits in the qureg
        m(int): the number of bits of the toffoli

    Returns:
        the circuit which describe the decomposition result
    """
    #print(n, m)
    circuit = controls + auxs + target
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
    
def OneDirtyAux(controls, aux, target):
    n = len(controls) + 2
    m1 = n // 2
    m2 = n - m1 - 1
    control1 = controls[0 : m1]
    auxs1 = controls[m1 : n - 2] + target
    target1 = aux
    control2 = controls[m1 : n - 1]
    auxs2 = controls[0 : m1]
    target2 =  target
    
    HalfDirtyAux(n, m1, control1, auxs1, target1)
    HalfDirtyAux(n, m2, control2, auxs2, target2)
    HalfDirtyAux(n, m1, control1, auxs1, target1)
    HalfDirtyAux(n, m2, control2, auxs2, target2)


class MCTLinearSimulationHalfDirtyAux(Synthesis):
    @classmethod
    def execute(cls, m, n):
        """ A linear simulation for Toffoli gate

        https://arxiv.org/abs/quant-ph/9503016 Lemma 7.2

        Implement a m-bit toffoli gate in a qureg with n qubit with linear complexity.

        If n ≥ 5 and m ∈ {3, . . . , ⌈n/2⌉} then (m+1)-Toffoli gate can be simulated
        by a network consisting of 4(m − 2) toffoli gates

        Returns:
            CompositeGate
        """
        if m > (n // 2) + (1 if n % 2 == 1 else 0):
            raise Exception("control bit cannot above ceil(n/2)")
        if m < 1:
            raise Exception("there must be at least one control bit")
        
        circuit = Circuit(n)
        controls = circuit(i for i in range(m))
        auxs = circuit(i for i in range(m,n-1))
        target = circuit(n)
        
        HalfDirtyAux(n, m, controls, auxs, target)
        
        return CompositeGate(circuit.gates)

class MCTLinearSimulationOneDirtyAux(Synthesis):
    @classmethod
    def execute(cls, n):
        """ A linear simulation for Toffoli gate

        https://arxiv.org/abs/quant-ph/9503016 Lemma 7.2

        Implement a m-bit toffoli gate in a qureg with n qubit with linear complexity.

        If n ≥ 5 and m ∈ {3, . . . , ⌈n/2⌉} then (m+1)-Toffoli gate can be simulated
        by a network consisting of 4(m − 2) toffoli gates

        Returns:
            CompositeGate
        """
        
        circuit = Circuit(n + 2)
        controls = circuit(i for i in range(n))
        aux = circuit(n)
        target = circuit(n + 1)
        
        OneDirtyAux(controls, aux, target)
        
        return CompositeGate(circuit.gates)
