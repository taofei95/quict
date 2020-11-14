#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/8/22 2:39 下午
# @Author  : Han Yu
# @File    : _algorithm.py

class Algorithm(object):
    @classmethod
    def run(cls, *pargs):
        """
        :param *pargs 参数列表
        :return: 返回参数
        """
        circuit = cls.__run__(*pargs)
        return circuit

    @staticmethod
    def __run__(*pargs):
        """
        需要其余算法改写
        :param circuit: *pargs 参数列表
        :return: 返回参数
        """
        return pargs[0]
