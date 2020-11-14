#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/2/11 10:33 下午
# @Author  : Han Yu
# @File    : _param2circuit.py

from QuICT.models import Circuit

class param2circuit(object):
    @classmethod
    def run(cls, *pargs):
        """
        :param *pargs 传入参数列表
        :return: 生成的电路
        """
        circuit = cls.__run__(*pargs)
        return circuit

    @staticmethod
    def __run__(*pargs):
        """
        需要其余算法改写
        :param *pargs: 参数列表
        :return: 生成的电路
        """
        return Circuit(pargs[0])
