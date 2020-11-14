#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/2/11 10:33 下午
# @Author  : Han Yu
# @File    : _param2param.py


class param2param(object):
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
