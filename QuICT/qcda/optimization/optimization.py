#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/8/22 2:39
# @Author  : Han Yu
# @File    : optimization.py

from QuICT.qcda.optimization import CommutativeOptimization


class Optimization(object):
    """ 
    In general, optimization algorithm means the algorithm which optimizes the input circuit
    to a better circuit, which is better in some aspects such as depth, size, T-count and so on.

    A Optimization means several qubit optimization methods, which would be executed sequentially.
    If the methods are not assigned, a default sequence will be given.
    """
    def __init__(self, methods=None):
        """
        Args:
            methods(list, optional): a list of used methods
        """
        if methods is not None:
            self.methods = methods
        else:
            self.methods = []
            commutative = CommutativeOptimization()
            self.methods.append(commutative)

    def execute(self, circuit):
        """
        Optimize the circuit with the given methods

        Args:
            circuit(CompositeGate/Circuit): the target CompositeGate or Circuit
        """
        for method in self.methods:
            circuit = method.execute(circuit)

        return circuit
