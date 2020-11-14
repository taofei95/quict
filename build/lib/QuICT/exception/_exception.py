#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/2/11 1:47 下午
# @Author  : Han Yu
# @File    : _exception.py

class TypeException(Exception):
    def __init__(self, type, now):
        """
        :param now: 错误索引list或者tuple
        """
        string = str("类型错误,应传入{},实际传入了{}".format(type, now))
        Exception.__init__(self, string)

class ConstException(Exception):
    def __init__(self, other):
        """
        :param other: 修改了不应修改的常量
        """
        Exception.__init__(self, "算法运行过程不应对{}进行修改".format(other))

class FrameworkException(Exception):
    def __init__(self, other):
        """
        :param other: 错误信息
        """
        Exception.__init__(self, "框架错误:{}".format(other))

class CircuitStructException(Exception):
    def __init__(self, other):
        """
        :param other: 错误信息
        """
        Exception.__init__(self, "非法电路:{}".format(other))

class QasmInputException(Exception):
    def __init__(self, other, line, file):
        """
        :param other: 错误信息
        """
        Exception.__init__(self, "Qasm输入错误:{} \n 错误行数:{} \n 错误文件:{}".format(other, line, file))
