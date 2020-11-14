#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/8/22 2:39 下午
# @Author  : Han Yu
# @File    : _optimization.py

from QuICT.models import Circuit, GateBuilder

class Optimization(object):
    @classmethod
    def run(cls, circuit : Circuit, *pargs, inplace=False):
        """
        :param circuit: 需变化电路
        :param inplace: 为真时,返回一个新的电路;为假时,修改原电路的门参数
        :return: inplace为真时,无返回值;为假时,返回新的电路,电路初值为0
        """
        circuit.const_lock = True
        gates = cls.__run__(circuit, *pargs)
        if isinstance(gates, Circuit):
            gates = gates.gates
        circuit.const_lock = False
        if inplace:
            circuit.set_flush_gates(gates)
        else:
            new_circuit = Circuit(len(circuit.qubits))
            new_circuit.set_flush_gates(gates)
            return new_circuit

    @staticmethod
    def __run__(circuit : Circuit, *pargs):
        """
        需要其余算法改写
        :param circuit: 需变化电路
        :return: 返回新电路门的数组
        """
        return circuit.gates
