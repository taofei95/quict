#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/8/22 2:39
# @Author  : Han Yu
# @File    : _mapping.py

from QuICT.core import Circuit

class Mapping(object):
    """ Mapping the logical qubits into reality device

    Note that all subClass must overloaded the function "_run".
    The overloaded of the function "run" is optional.

    """
    @classmethod
    def run(cls, circuit: Circuit, *pargs, inplace=False):
        """
        Args:
            circuit(Circuit): the circuit waited to be mapped, contained topology
            *pargs: other parameters
            inplace(bool): return a new circuit if it is true,
                otherwise change the origin circuit
        """
        circuit.const_lock = True
        gates = cls._run(circuit, *pargs)
        circuit.const_lock = False
        if inplace:
            circuit.set_exec_gates(gates)
        else:
            new_circuit = Circuit(len(circuit.qubits))
            new_circuit.set_exec_gates(gates)
            return new_circuit

    @staticmethod
    def _run(*pargs):
        """ should be overloaded by subClass

        """
        return pargs[0]
