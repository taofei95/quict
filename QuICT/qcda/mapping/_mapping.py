#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/8/22 2:39 下午
# @Author  : Han Yu
# @File    : _mapping.py

from typing import List, Dict,Tuple,Union
from abc import abstractmethod, abstractclassmethod, abstractstaticmethod
from table_based_mcts import *
from QuICT.core.circuit import * 
from QuICT.core.exception import *
from QuICT.core.gate import *
from QuICT.core.layout import *
from qubit_mapping.qubit_mapping import QubitMapping as qm

class Mapping(object):
    @abstractclassmethod
    def run(cls)->Circuit:
        pass

    @abstractstaticmethod
    def _run(cls)->Circuit:
        pass

class  Mapping_1D(Mapping):
    @classmethod
    def run(cls, circuit: Circuit, num: int, init_mapping: List[int], method: str = "greedy_search", inplace: bool = False) -> Circuit:
        """Mapping the logical circuit to a physical device in linear neareast neighbor architecture.
        Args:
            circuit: The input circuit that needs to be mapped into a 1D physical architecture 
            num: The number of physical qubits
            #. ``Layout`` instance:  
            #. List[index]:  index: logical qubit -> List[index]:physical qubit
                * 4-qubit device init_mapping: [ 3, 2, 0, 1 ]
                  logical qubit -> physical qubit
                  0         3
                  1         2
                  2         0
                  3         1

            method: The algorithm used to transform the logical quantum circuit to hardware-compliant circuit.
            inplace: Indicate wether the function returns a new circuit or modifies the circuit passed to the function. 
        Return:  
            the hardware-compliant circuit after mapping  and the init_mapping for the circuit.
        Raise:

        """
        circuit.const_lock = True
        num_logic = len(circuit.qubits)
        if num_logic > num:
            raise Exception("There are not enough physical qubits to excute the circuit")

        gates = cls._run(circuit = circuit, num=num, init_mapping = init_mapping ,method = method)
        
        circuit.const_lock = False
        circuit.qubits
        if inplace:
            circuit.set_flush_gates(gates)
        else:
            new_circuit = Circuit(num)
            new_circuit.extend(gates)
            return new_circuit
    
    @staticmethod
    def _run(circuit : Circuit, method : str, init_mapping: List[int], num : int) -> List[BasicGate]:
        """
        """
        circuit_ori = []
        for g in circuit.gates:
            temp = {}
            if g.controls + g.targets == 2:
                if g.controls == 1:
                    temp['ctrl'] = g.carg
                    temp['tar'] = g.targ
                elif g.controls == 2:
                    temp['ctrl'] = g.cargs[0]
                    temp['tar'] = g.cargs[1]
                else:
                    temp['ctrl'] = g.targs[0]
                    temp['tar'] = g.targs[1]
                temp['type'] = 2
            elif g.controls + g.targets == 1:
                temp['ctrl'] = g.targ
                temp['tar'] = g.targ
                temp['type'] = 1
            else:
                raise TypeException("two-qubit gate or single qubit gate", "the gate acting on more than two qubits")        
            temp['name'] = g.type()

            circuit_ori.append(temp)

        trans = qm(circuit = circuit_ori, num =num, method = method )

        circuit_trans, mapping = trans.get_circuit()
        init_mapping[:] = mapping
        gates = []
        index = 0

        for gate in circuit_trans:
            if gate['name'] == 30:
                GateBuilder.setGateType(GATE_ID['Swap'])
                GateBuilder.setTargs([gate['ctrl'], gate['tar']])
                GateBuilder.setCargs([])
                GateBuilder.setPargs([])
            else:
                GateBuilder.setGateType(gate['name'])
                if gate['type'] == 2:
                    if circuit.gates[index].targets == 2:
                        GateBuilder.setTargs([gate['ctrl'], gate['tar']])
                        GateBuilder.setCargs([])
                    elif circuit.gates[index].targets == 1:
                        GateBuilder.setCargs(gate['ctrl']) 
                        GateBuilder.setTargs(gate['tar'])
                    else:
                        GateBuilder.setCargs([gate['ctrl'], gate['tar']])
                        GateBuilder.setTargs([])  
                elif gate['type'] == 1:
                    GateBuilder.setTargs(gate['tar'])
                    GateBuilder.setCargs([])

                GateBuilder.setPargs(circuit.gates[index].pargs)
                index += 1
            g=GateBuilder.getGate()
            gates.append(GateBuilder.getGate())
        return gates


class Mapping_NISQ(Mapping):  
    @classmethod
    def run(cls, circuit: Circuit, num: int,  init_mapping: List[int], layout, inplace: bool = False,parameter: Dict = {})-> Circuit:
        """Mapping the logical circuit to a NISQ device.
        Args:
            circuit: The input circuit that needs to be mapped to a NISQ device.
            layout: The coupling graph of the NISQ devices.
            num: The number of physical qubits 
            init_mapping: Initial position of logical qubits on physical qubits. It might be changed by the mapping method.
            #. ``Layout`` instance:  
            #. List[index]:  index: logical qubit -> List[index]:physical qubit
                * 4-qubit device init_mapping: [ 3, 2, 0, 1 ]
                  logical qubit -> physical qubit
                  0         3
                  1         2
                  2         0
                  3         1
            parameter: The parameters that might  be used in the mapping method.
            inplace: Indicate wether the function returns a new circuit or modifies the circuit passed to the function. 
        Return:  
            the hardware-compliant circuit after mapping  and the init_mapping for the circuit.
        Raise:

        """
        circuit.const_lock = True
        num_logic = len(circuit.qubits)
        num_physic = num
    
        if num_logic > num_physic:
            raise Exception("There are not enough physical qubits to excute the circuit")

        gates = cls._run(circuit = circuit, layout = layout, init_mapping = init_mapping , parameter = parameter)
        
        circuit.const_lock = False
        circuit.qubits
        if inplace:
            circuit.set_flush_gates(gates)
        else:
            new_circuit = Circuit(num)
            new_circuit.extend(gates)
            return new_circuit
        
    @staticmethod
    def _run(circuit: Circuit, init_mapping: List[int], layout, parameter: Dict = {})-> Circuit:
        """Use Monte Carlo tree search algorithm to solve the qubit mapping problem
        """

        mcts_tree = TableBasedMCTS(**parameter)
        mcts_tree.search(logical_circuit = circuit, init_mapping = init_mapping, coupling_graph = layout)
        return mcts_tree.physical_circuit


        
