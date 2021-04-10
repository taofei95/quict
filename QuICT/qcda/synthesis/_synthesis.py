#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/8/22 2:39
# @Author  : Han Yu
# @File    : _synthesis.py

from typing import *

import numpy as np

from QuICT.core.exception import TypeException
from QuICT.core import Circuit, Qubit, Qureg, CompositeGate


class Synthesis(object):
    """ synthesis some oracle into BasicGate

    In general, these BasicGates are one-qubit gates and two-qubit gates.

    in _gate.py, we define a class named gateModel, which is similar with this class.
    The difference of them is that gateModel describes simple and widely accepted to a certain extent.
    And this class describes harder synthesis method, some of which may have a lot room to improve.

    When users use the synthesis method, it should be similar with class gateModel. So the attributes and
    API is similar with class gateModel.

    Note that all subClass must overloaded the function "__call__".
    """

    def __init__(self, function: Callable):
        # TODO: annotations
        # self.__call__.annotations__ = function.__annotations__
        self._synthesisFunction = function

    def __call__(self, *pargs, **kwargs) -> CompositeGate:
        """

        Args:
            *pargs: parameters
            **kwargs: key word arguments
        Returns:
            CompositeGate: the list of results
        """
        return self._synthesisFunction(*pargs, **kwargs)

    def _synthesisFuncion(*pargs):
        """

        Args:
            *pargs: parameters
        Returns:
            CompositeGate: the list of results
        """
        raise Exception('"__call__" function must be overloaded')
