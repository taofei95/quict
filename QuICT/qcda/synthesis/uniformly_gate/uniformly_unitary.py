#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/12/28 12:29 上午
# @Author  : Han Yu
# @File    : uniformly_unitary.py

from .._synthesis import Synthesis
from QuICT.core import GateBuilder, GATE_ID

def uniformlyUnitary(low, high, z):
    """ synthesis uniformlyUnitary gate, bits range [low, high)
    Args:
        low(int): the left range low
        high(int): the right range high
        z(list<int>): the list of angle y
    Returns:
        the synthesis result
    """
    return []

class uniformlyUnitaryGate(Synthesis):
    """ uniformUnitaryGate

    http://cn.arxiv.org/abs/quant-ph/0504100v1 Fig4 b)
    """

    def __call__(self, unitary_list):
        """
        Args:
            unitary_list(list<np.ndarray>): the angles of Unitary Gates
        Returns:
            uniformlyUnitaryGate: model filled by the parameter angle_list.
        """
        self.pargs = unitary_list
        return self

    def build_gate(self):
        """ overloaded the function "build_gate"

        """
        n = self.targets
        if 1 << (n - 1) != len(self.pargs):
            raise Exception("the number of parameters unmatched.")
        return uniformlyUnitary(0, n, self.pargs)
