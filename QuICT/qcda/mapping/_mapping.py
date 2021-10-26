#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/8/22 2:39
# @Author  : Han Yu
# @File    : _mapping.py

from abc import ABC, abstractclassmethod


class Mapping(ABC):
    """ Mapping the logical qubits into reality device

    Note that all subclass must overload the function "execute".
    """
    @abstractclassmethod
    def execute(cls, *args, **kwargs):
        """ Mapping process to be implemented

        Args:
            *args: arguments
            **kwargs: keyword arguments

        Raises:
            NotImplementedError: If it is not implemented.
        """
        raise NotImplementedError
