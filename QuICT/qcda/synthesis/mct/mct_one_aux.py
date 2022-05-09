#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/12 11:37
# @Author  : Han Yu
# @File    : MCT_one_aux.py

from QuICT.core import *
from QuICT.core.gate import *
from .mct_linear_simulation import MCTLinearHalfDirtyAux


class MCTOneAux(object):
    def execute(self, qubit):
        """ Decomposition of n-qubit Toffoli gates with one ancillary qubit and linear circuit complexity

        He Y, Luo M X, Zhang E, et al.
        Decompositions of n-qubit Toffoli gates with linear circuit complexity[J].
        International Journal of Theoretical Physics, 2017, 56(7): 2350-2361.

        Args:
            qubit(int): the number of used qubit, which is (n + 1) for n-qubit Toffoli gates

        Returns:
            CompositeGate: the result of Decomposition
        """
        n = qubit - 1
        gates = CompositeGate()
        qubit_list = list(range(n + 1))
        with gates:
            if n == 3:
                CCX & qubit_list[:3]
                return gates
            elif n == 2:
                CX & qubit_list[:2]
                return gates
            if n % 2 == 1:
                k1 = n // 2 + 1
            else:
                k1 = n // 2
            k2 = n // 2 - 1

            yet_another_qubit_list = qubit_list[k1:k1 + k2 + 1] + qubit_list[:k1] + [qubit_list[-1]]
            yet_m = k2 + 1
            yet_n = n + 1
            if yet_m > (yet_n // 2) + (1 if yet_n % 2 == 1 else 0):
                raise Exception("control bit cannot above ceil(n/2)")
            if yet_m < 1:
                raise Exception("there must be at least one control bit")
            yet_controls = [yet_another_qubit_list[i] for i in range(yet_m)]
            yet_auxs = [yet_another_qubit_list[i] for i in range(yet_m, yet_n - 1)]
            yet_target = yet_n - 1

            MCT_half_dirty = MCTLinearHalfDirtyAux()
            half_dirty_gates = MCT_half_dirty.execute(k1, n + 1)
            yet_gates = MCT_half_dirty.assign_qubits(yet_n, yet_m, yet_controls, yet_auxs, yet_target)

            half_dirty_gates | gates
            H & qubit_list[-2]
            S & qubit_list[-1]
            yet_gates | gates
            S_dagger & qubit_list[-1]
            half_dirty_gates | gates
            S & qubit_list[-1]
            yet_gates | gates

            H & qubit_list[-2]
            S_dagger & qubit_list[-1]

        return gates
