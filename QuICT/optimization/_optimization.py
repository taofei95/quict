#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/8/22 2:39 下午
# @Author  : Han Yu
# @File    : _optimization.py

from QuICT.models import Circuit

class Optimization(object):
    """ SuperClass of all optimization algorithm

    In general, optimization algorithm means the algorithm which optimizes the input circuit
    to a better circuit, which is better is some aspect such as depth, size, T-count and so on

    Note that all subClass must overloaded the function "_run".
    The overloaded of the function "__run__" is optional.

    """

    @classmethod
    def run(cls, circuit : Circuit, *pargs, inplace = False):
        """ optimize the circuit

        Args:
            circuit(Circuit): the circuit to be optimize
            *pargs:           the parameters to be filled
            inplace(bool):    change the old circuit if it is true, otherwise create a new circuit

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
        """ private function to solve the problem
        Args:
            circuit(Circuit): the circuit to be optimize
            *pargs:           the parameters to be filled

        """
        print(*pargs)
        return circuit.gates
