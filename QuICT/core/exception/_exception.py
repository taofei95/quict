#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/2/11 1:47 下午
# @Author  : Han Yu
# @File    : _exception.py

class TypeException(Exception):
    """ Exception that the type of parameter is error.

    """
    def __init__(self, type, now):
        """
        Args:
            type: the type should passed in
            now: the type actually passed in
        """
        string = str(f"type error,{type} should be passed in, {now} is passed in actually")
        Exception.__init__(self, string)

class ConstException(Exception):
    """ Exception that change the variable which shouldn't be

    """
    def __init__(self, other):
        """
        Args:
            other: the variable
        """
        Exception.__init__(self, f"the running process shouldn't change the {other}")

class FrameworkException(Exception):
    """ Exception that framework may have error

    """
    def __init__(self, other):
        """
        Args:
            other: error information
        """
        Exception.__init__(self, f"framework error:{other}")

class CircuitStructException(Exception):
    """ Exception that circuit struct may have error

    """
    def __init__(self, other):
        """
        Args:
            other: error information
        """
        Exception.__init__(self, f"circuit struct error:{other}")

class QasmInputException(Exception):
    """ Exception that Qasm Input may have error

    """
    def __init__(self, other, line, file):
        """
        Args:
            other: error type
            line: error line
            file: error file
        """
        Exception.__init__(self, "Qasm error:{} \n in line:{} \n error file:{}".format(other, line, file))

class IndexLimitException(Exception):
    """ Exception that out of index

    """
    def __init__(self, wire, try_index):
        """
        Args:
            wire: the index range
            try_index: index passed in
        """
        Exception.__init__(self, f"out of index: the index range is [0, {wire}),but try to get{try_index}")

class IndexDuplicateException(Exception):
    """ Exception that duplicate indexes

    """
    def __init__(self, other):
        """
        Args:
            other: the indexes passed in
        :param other:
        """
        Exception.__init__(self, f"duplicate indexes: {other}")
