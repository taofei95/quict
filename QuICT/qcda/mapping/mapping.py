#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/8/22 2:39
# @Author  : Han Yu
# @File    : mapping.py

from QuICT.qcda.mapping import MCTSMapping


class Mapping(object):
    """ Mapping the logical qubits into reality device

    A Mapping means several qubit mapping methods, which would be executed sequentially.
    If the methods are not assigned, a default sequence will be given.
    """
    def __init__(self, layout=None, methods=None):
        """
        Args:
            layout(Layout): topology of the target physical device
            methods(list, optional): a list of used methods
        """
        assert layout is not None, ValueError('No Layout provided for Mapping')

        if methods is not None:
            self.methods = methods
        else:
            self.methods = []
            MCTS = MCTSMapping(layout=layout, init_mapping_method='anneal')
            self.methods.append(MCTS)

    def execute(self, circuit):
        """
        Map the circuit to the given Layout with the given methods

        Args:
            circuit(CompositeGate/Circuit): the target CompositeGate or Circuit
        """
        for method in self.methods:
            circuit = method.execute(circuit)

        return circuit
