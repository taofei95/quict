#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/8/22 2:39
# @Author  : Han Yu
# @File    : _optimization.py

from abc import ABC, abstractclassmethod


class Optimization(ABC):
    """ SuperClass of all optimization algorithm

    In general, optimization algorithm means the algorithm which optimizes the input circuit
    to a better circuit, which is better in some aspects such as depth, size, T-count and so on.
    Note that all subclass must overload the function "execute".
    """
    @abstractclassmethod
    def execute(cls, *args, **kwargs):
        """ Optimization process to be implemented

        Args:
            *args: arguments
            **kwargs: keyword arguments

        Raises:
            NotImplementedError: If it is not implemented.
        """
        raise NotImplementedError
