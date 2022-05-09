#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/12 6:33
# @Author  : Han Yu
# @File    : MCT_Linear_Simulation.py

from QuICT.core import *
from QuICT.core.gate import *


def half_dirty_aux(gates, n, m, controls: list, auxs: list, target: int):
    """

    Args:
        n(int): the number of qubits in the qureg
        m(int): the number of bits of the toffoli

    Returns:
        the circuit which describe the decomposition result
    """
    with gates:
        circuit = controls + auxs + [target]
        if m == 1:
            CX & [circuit[0], circuit[n - 1]]
        elif m == 2:
            CCX & [circuit[0], circuit[1], circuit[n - 1]]
        else:
            for i in range(m, 2, -1):
                CCX & [circuit[i - 1], circuit[n - 1 - (m - i + 1)], circuit[n - 1 - (m - i)]]
            CCX & [circuit[0], circuit[1], circuit[n - m + 1]]
            for i in range(3, m + 1):
                CCX & [circuit[i - 1], circuit[n - 1 - (m - i + 1)], circuit[n - 1 - (m - i)]]

            for i in range(m - 1, 2, -1):
                CCX & [circuit[i - 1], circuit[n - 1 - (m - i + 1)], circuit[n - 1 - (m - i)]]
            CCX & [circuit[0], circuit[1], circuit[n - m + 1]]
            for i in range(3, m):
                CCX & [circuit[i - 1], circuit[n - 1 - (m - i + 1)], circuit[n - 1 - (m - i)]]


class MCTLinearHalfDirtyAux(object):
    @staticmethod
    def execute(m, n):
        """ A linear simulation for Toffoli gate

        https://arxiv.org/abs/quant-ph/9503016 Lemma 7.2

        Implement a m-bit toffoli gate in a qureg with n qubit with linear complexity.

        If n ≥ 5 and m ∈ {3, . . . , ⌈n/2⌉} then (m+1)-Toffoli gate can be simulated
        by a network consisting of 4(m - 2) toffoli gates

        Returns:
            CompositeGate
        """
        if m > (n // 2) + (1 if n % 2 == 1 else 0):
            raise Exception("control bit cannot above ceil(n/2)")
        if m < 1:
            raise Exception("there must be at least one control bit")

        controls = [i for i in range(m)]
        auxs = [i for i in range(m, n - 1)]
        target = n - 1

        gates = CompositeGate()
        with gates:
            half_dirty_aux(gates, n, m, controls, auxs, target)
        return gates


class MCTLinearOneDirtyAux(object):
    def execute(self, n):
        """ A linear simulation for Toffoli gate

        https://arxiv.org/abs/quant-ph/9503016 Corollary 7.4

        Implement an n-bit toffoli gate in a qureg with n + 2 qubits with linear complexity.

        On an n-bit circuit, an (n-2)-bit toffoli gate can be simulated by
        8(n-5) CCX gates, with 1 bit dirty ancilla.

        Args:
            n(int): the number of used qubit, which is (n + 2) for n-qubit Toffoli gates

        Returns:
            CompositeGate
        """
        if n < 3:
            raise Exception("there must be at least one control bit")

        controls = list(range(n - 2))
        target = n - 2
        aux = n - 1     # this is a dirty ancilla

        gates = CompositeGate()
        with gates:
            n = len(controls) + 2
            if n == 5:
                CCX & [controls[0], controls[1], aux]
                CCX & [controls[2], aux, target]
                CCX & [controls[0], controls[1], aux]
                CCX & [controls[2], aux, target]
                return gates
            if n == 4:
                CCX & [controls[0], controls[1], target]
                return gates
            if n == 3:
                CX & [controls[0], target]
                return gates
            # n > 5
            m1 = n // 2
            m2 = n - m1 - 1
            control1 = controls[0: m1]
            auxs1 = controls[m1: n - 2] + [target]
            target1 = aux
            control2 = controls[m1: n - 2] + [aux]
            auxs2 = controls[0: m1]
            target2 = target

            # half_dirty_aux(n, m1, control1, auxs1, target1)
            if m2 == 2:  # n == 6
                half_dirty_aux(gates, n, m1, control1, auxs1, target1)
                CCX & [control2[0], control2[1], target2]
                half_dirty_aux(gates, n, m1, control1, auxs1, target1)
                CCX & [control2[0], control2[1], target2]
            else:
                half_dirty_aux(gates, n, m1, control1, auxs1, target1)
                half_dirty_aux(gates, n, m2, control2, auxs2, target2)
                half_dirty_aux(gates, n, m1, control1, auxs1, target1)
                half_dirty_aux(gates, n, m2, control2, auxs2, target2)

        return gates
