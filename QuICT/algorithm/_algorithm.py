#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/8/22 2:39 下午
# @Author  : Han Yu
# @File    : _algorithm.py

class Algorithm(object):
    """ quantum algorithm which should run in the quantum hardware

    the subClass should overloaded the function _run,
    the overloaded of run is optional

    """
    @classmethod
    def run(cls, *pargs):
        """
        Args:
            *pargs: parameters which needed by the algorithm.
        """
        result = cls._run(*pargs)
        return result

    @staticmethod
    def _run(*pargs):
        """
        Args:
            *pargs: parameters which needed by the algorithm.
        """
        return pargs[0]
