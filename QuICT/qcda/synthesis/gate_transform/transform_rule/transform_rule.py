#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/3/20 5:11 下午
# @Author  : Han Yu
# @File    : TransformRule.py

import random

import numpy as np

from QuICT.core.gate import BasicGate
from QuICT.core.gate.gate_builder import GATE_TYPE_TO_CLASS


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
        if isinstance(source, BasicGate):
            source = source.type
        self.__source = source

    @property
    def target(self):
        if self.__target is None:
            raise Exception("the transform rule do not have target.")
        return self.__target

    @target.setter
    def target(self, target):
        if isinstance(target, BasicGate):
            target = target.type
        self.__target = target

    def __init__(self, funtion, source=None, target=None):
        """ initial transform rule

        Args:
            funtion(function): the function that execute the transformation
            source(int/BasicGate): the source gate in situation 1
            target(int/BasicGate): the target gate in situation 1
        """
        self.transform = funtion
        self.source = source
        self.target = target

    def check_equal(self, ignore_phase=True, eps=1e-7):
        """ check whether the rule is true

        Args:
            ignore_phase(bool): whether ignore the global phase
            eps(float): tolerance of precision

        Returns:
            bool: whether the rule is ture
        """
        if not self.source:
            raise Exception("it is used for two qubit rules.")
        gate = GATE_TYPE_TO_CLASS[self.source]().copy()
        # gate.affectArgs = [i for i in range(gate.targets + gate.controls)]
        gate.cargs = [i for i in range(gate.controls)]
        gate.targs = [i for i in range(gate.controls, gate.controls + gate.targets)]
        gate.pargs = [random.random() * 2 * np.pi for _ in range(gate.params)]
        compositeGate = self.transform(gate)
        return compositeGate.equal(gate, ignore_phase=ignore_phase, eps=eps)
