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
    if n == 5:
        CCX | (controls[0], controls[1], aux)
        CCX | (controls[2], aux, target)
        CCX | (controls[0], controls[1], aux)
        CCX | (controls[2], aux, target)
        return
    if n == 4:
        CCX | (controls[0], controls[1], target)
        return
    if n == 3:
        CX | (controls, target)
        return
    # n > 5
    m1 = n // 2
    m2 = n - m1 - 1
    control1 = controls[0 : m1]
    auxs1 = controls[m1 : n - 2] + target
    target1 = aux
    control2 = controls[m1 : n - 2] + aux
    auxs2 = controls[0 : m1]
    target2 = target
    
    HalfDirtyAux(n, m1, control1, auxs1, target1)
    if m2 == 2: # n == 6
        HalfDirtyAux(n, m1, control1, auxs1, target1)
        CCX | (control2[0], control2[1], target2)
        HalfDirtyAux(n, m1, control1, auxs1, target1)
        CCX | (control2[0], control2[1], target2)
    else:
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
        controls = circuit([i for i in range(m)])
        auxs = circuit([i for i in range(m, n - 1)])
        target = circuit(n - 1)
        
        HalfDirtyAux(n, m, controls, auxs, target)
        
        return CompositeGate(circuit.gates)

class MCTLinearSimulationOneDirtyAux(Synthesis):
    @classmethod
    def execute(cls, n):
        """ A linear simulation for Toffoli gate

        https://arxiv.org/abs/quant-ph/9503016 Corollary 7.4

        Implement an n-bit toffoli gate in a qureg with n + 2 qubits with linear complexity.

        On an (n+2)-bit circuit, an n-bit toffoli gate can be simulated by 
        8(n-5) CCX gates, with 1 bit dirty ancilla.

        Returns:
            CompositeGate
        """
        if n < 1:
            raise Exception("there must be at least one control bit")

        circuit = Circuit(n + 2)
        controls = circuit([i for i in range(n)])
        aux = circuit(n)       # this is a dirty ancilla
        target = circuit(n + 1)
        
        OneDirtyAux(controls, aux, target)
        
        return CompositeGate(circuit.gates)
