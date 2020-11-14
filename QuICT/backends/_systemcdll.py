#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/2/15 10:43 下午
# @Author  : Han Yu
# @File    : _systemcdll.py.py

import ctypes
import os
import platform
import numpy as np
"""
静态库
"""
class SystemCdll(object):
    """
    类的属性
    """
    @property
    def quick_operator_cdll(self):
        """
        :return: 懒加载门库
        """
        if self.__quick_operator_cdll is None:
            sys = platform.system()
            path = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "quick_operator_cdll.so"
            self.__quick_operator_cdll = ctypes.cdll.LoadLibrary(path)
        return self.__quick_operator_cdll

    def __init__(self):
        self.__quick_operator_cdll = None

systemCdll = SystemCdll()
