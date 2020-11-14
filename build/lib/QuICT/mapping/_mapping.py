#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/8/22 2:39 下午
# @Author  : Han Yu
# @File    : _mapping.py

from QuICT.models import Circuit

class Mapping(object):
    @classmethod
    def run(cls, circuit: Circuit, inplace=False):
        """
        :param circuit: 需变化电路
        :param inplace: 为真时,返回一个新的电路;为假时,修改原电路的门参数
        :return: inplace为真时,无返回值;为假时,返回新的电路,电路初值为0
        """
        circuit.const_lock = True
        gates = cls.__run__(circuit)
        circuit.const_lock = False
        if inplace:
            circuit.set_flush_gates(gates)
        else:
            new_circuit = Circuit(len(circuit.qubits))
            new_circuit.set_flush_gates(gates)
            return new_circuit

    @staticmethod
    def __run__(*pargs):
        """
        需要其余算法改写
        :param *pargs 参数列表
        :return: 返回新电路
        """
        return pargs[0]
