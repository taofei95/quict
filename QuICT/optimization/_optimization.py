#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/8/22 2:39 下午
# @Author  : Han Yu
# @File    : _optimization.py

from QuICT.models import Circuit

class Optimization(object):
    """ The superClass of Optimization algorithm



    """
    @classmethod
    def run(cls, circuit : Circuit, *pargs, inplace=False):
        """ run the algorithm
        circuit(Circuit): the circuit to be optimize
        inplace(bool):
            if true, change the origin circuit.
            otherwise, return a new circuit with origin circuit constant.
            default to be false
        *pargs: other parameters
        """
        circuit.const_lock = True
        gates = cls._run(circuit, *pargs)
        if isinstance(gates, Circuit):
            gates = gates.gates
        circuit.const_lock = False
        if inplace:
            circuit.set_flush_gates(gates)
        else:
            new_circuit = Circuit(len(circuit.qubits))
            new_circuit.set_flush_gates(gates)
            return new_circuit

    @staticmethod
    def _run(circuit : Circuit, *pargs):
        """
        circuit(Circuit): the circuit to be optimize
        *pargs: other parameters
        """
        print(*pargs)
        return circuit.gates
