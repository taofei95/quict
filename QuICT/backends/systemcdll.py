#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/2/15 10:43
# @Author  : Han Yu
# @File    : _systemcdll.py.py

import ctypes
import os


class SystemCdll(object):
    """ calculation module that coded by C++ with Inter tbb parallel library

    Attributes:
        quick_operator_cdll(_DLLT): the lazy-load library

    """
    @property
    def quick_operator_cdll(self):
        if self.__quick_operator_cdll is None:
            path = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "quick_operator_cdll.so"
            self.__quick_operator_cdll = ctypes.cdll.LoadLibrary(path)
        return self.__quick_operator_cdll

    def __init__(self):
        self.__quick_operator_cdll = None


systemCdll = SystemCdll()
