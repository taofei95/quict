#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/8/22 2:39
# @Author  : Han Yu
# @File    : _synthesis.py

from abc import ABC, abstractclassmethod


class Synthesis(ABC):
    """ Synthesize gates with some basic gates

    In general, these basic gates are one-qubit gates and two-qubit gates.
    Note that all subclass must overload the function "execute".
    """
    @abstractclassmethod
    def execute(cls, *args, **kwargs):
        """ Synthesis process to be implemented

        Args:
            *args: arguments
            **kwargs: keyword arguments

        Raises:
            NotImplementedError: If it is not implemented.
        """
        raise NotImplementedError
