#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/8/22 2:39 
# @Author  : Han Yu
# @File    : _mapping.py

from QuICT.core.layout import *
from .mcts import *
from .utility import *


class Mapping(object):
    @classmethod
    def run(cls, circuit: Circuit, layout: Layout, init_mapping: List[int] = None, 
            inplace: bool = False, **parameter) -> Circuit:
        """Mapping the logical circuit to a NISQ device.
        Args:
            circuit: The input circuit that needs to be mapped to a NISQ device.
            layout: The physical layout of the NISQ devices.
            init_mapping: Initial position of logical qubits on physical qubits. 
                The argument is optional. If not given, it will be determined by init_mapping method. 
                A simple Layout instance is shown as follow:  
                    index: logical qubit -> List[index]:physical qubit
                        4-qubit device init_mapping: [ 3, 2, 0, 1 ]
                            logical qubit -> physical qubit
                            0         3
                            1         2
                            2         0
                            3         1
            inplace: Indicate wether the algorithm returns a new circuit or modifies the original circuit in place. 
            parameter: The parameters that might  be used in the mapping method.
        Return:  
            the hardware-compliant circuit after mapping.

        """
        circuit.const_lock = True
        num = layout.qubit_number
        gates = cls._run(circuit=circuit, layout=layout, init_mapping=init_mapping, **parameter)

        circuit.const_lock = False
        circuit.qubits
        if inplace:
            circuit.set_exec_gates(gates)
        else:
            new_circuit = Circuit(num)
            new_circuit.set_exec_gates(gates)
            return new_circuit

    @staticmethod
    def _run(circuit: Circuit, init_mapping: List[int], layout: Layout,  **parameter) -> List[BasicGate]:
        """      
        """
        pass