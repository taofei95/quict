#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/4/27 2:02 下午
# @Author  : Han Yu
# @File    : _simulation

import numpy as np

from QuICT.core import *
from QuICT.ops.linalg import multiply

class BasicSimulator(object):

    @staticmethod
    def pretreatment(self, circuit):
        """
        :param
            circuit(Circuit): the circuit needs pretreatment.
        :return:
            CompositeGate: the gates after pretreatment
        """
        gates = CompositeGate()
        gateSet = [np.identity(2, dtype=np.complex) for _ in range(circuit.circuit_width())]
        tangle = [i for i in range(circuit.circuit_width())]
        for gate in circuit.gates:
            if gate.target + gate.controls >= 3:
                raise Exception("only support 2-qubit gates and 1-qubit gates.")
            if gate.target + gate.controls == 1:
                target = gate.target
                if tangle[target] == target:
                    gateSet[target] = multiply(gate.matrix, gateSet[target])
                elif tangle[target] < target:
                    gateSet[target] = multiply(np.kron(np.identity(2, dtype=np.complex), gate.matrix), gateSet[target])
                else:
                    gateSet[target] = multiply(np.kron(gate.matrix, np.identity(2, dtype=np.complex)), gateSet[target])
            else:
                target1 = gate.target1
                target2 = gate.target2
                if tangle[target1] == target2:
                    pass

        return gates
