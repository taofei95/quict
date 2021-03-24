#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/3/20 5:11 下午
# @Author  : Han Yu
# @File    : TransformRule.py

from QuICT.core import BasicGate, GATE_ID

class TransformRule(object):
    """ a class describe a transform rule

    the transform rule can be:
        1. a fix rule from one two-qubit gate to another with some one qubits,
           source gate id and target gate id should be assigned in this situation.
        2. a rule decomposition SU(2) into some instruction set
        3. a rule decomposition SU(4) into some instruction set

    Attributes:
        function(function): the function that execute the transformation
        source(int): the id of source gate in situation 1
        target(int): the id of target gate in situation 1
    """

    @property
    def transform(self):
        return self.__transform

    @transform.setter
    def transform(self, function):
        self.__transform = function

    @property
    def source(self):
        if self.__source is None:
            raise Exception("the transform rule do not have source.")
        return self.__source

    @source.setter
    def source(self, source):
        self.__source = source

    @property
    def target(self):
        if self.__target is None:
            raise Exception("the transform rule do not have target.")
        return self.__target

    @target.setter
    def target(self, target):
        self.__target = target

    def __init__(self, funtion, source = None, target = None):
        """ initial transform rule

        Args:
            funtion(function): the function that execute the transformation
            source(int/BasicGate): the source gate in situation 1
            target(int/BasicGate): the target gate in situation 1
        """
        self.transform = funtion
        self.source = source
        self.target = target
