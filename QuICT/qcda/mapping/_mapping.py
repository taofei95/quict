#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/8/22 2:39 下午
# @Author  : Han Yu
# @File    : _mapping.py

from lib.qubit_mapping  import QubitMapping as qm

from table_based_mcts import *
from QuICT.core.circuit import * 
from QuICT.core.exception import *
from QuICT.core.gate import *
from typing import List, Dict,Tuple,Union
class  Mapping(object):
    @classmethod
    def run(cls, circuit: Circuit, num: int, init_mapping: List[int], is_lnn: bool = True,
            method: str = "greedy_search", inplace: bool = False, parameter:Dict = {}) -> Tuple[Circuit, List[int]]:
        """
        Args:
            circuit: The input circuit that needs to be mapped into a 1D physical architecture 
            num: The number of physical qubits
            init_mapping: Initial position of logical qubits on physical qubits. It might be changed by the mapping method.
            #. ``Layout`` instance:  
            #. List  index : physic -> init_mapping[index]:logic
                * 4-qubit device init_mapping: [ 3, 2, 0, 1 ]
                  physic -> logic
                  0         3
                  1         2
                  2         0
                  3         1

            is_lnn: A Boolean varaible to indicate whether the physical device is a 1D chain or linear-neareast-neighbour architecture 
            method: The algorithm used to transform the logical quantum circuit to hardware-compliant circuit.
            inplace: 
        Return:  
            the hardware-compliant circuit after mapping  and the init_mapping for the circuit
        Raise:

        """
        circuit.const_lock = True
        num_logic = len(circuit.qubits)
        if num_logic > num:
            raise Exception("There are not enough phyical qubits to excute the circuit")

        if is_lnn is True:
            gates = cls._mapping_1D(circuit = circuit, num=num, init_mapping = init_mapping ,method = method)
            # for i in init_mapping:
            #     print(i)
        else:
            gates = cls._mapping_NISQ(circuit = circuit, num=num, init_mapping = init_mapping, method = method)
        
       
        circuit.const_lock = False
        circuit.qubits
        if inplace:
            circuit.set_flush_gates(gates)
        else:
            new_circuit = Circuit(num)
            new_circuit.extend(gates)
            return new_circuit
    
    @staticmethod
    def _mapping_1D(circuit : Circuit, method : str, init_mapping: List[int], num : int) -> Tuple[List[BasicGate], List[int]]:
        """
        Args:
            circuit: the input circuit that needs to be mapped into a 1D physical architecture 
            method: The algorithm used to transform the logical quantum circuit to hardware-compliant circuit.
            num: the number of physical qubits
        Return:
            gates:  the hardware-compliant circuit after mapping 
            init_mapping: the optimal initial position of logical qubits on physical qubits found by the mapping method 
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
            #print(g.type().value)
            circuit_ori.append(temp)
       # for w in circuit_ori:
       #     print(w)
        #print("qm1")
        #print(num)
        trans = qm(circuit = circuit_ori, num =num, method = method )
        #print("qm")

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
                # print(circuit.gates[index].type().value, end = ' ')
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
            # print(gate['ctrl'], end = ' ')
            # print(gate['tar'], end=' ')
            # print(gate['name'])
            # print(GateBuilder.targs, end=' ')
            # print(GateBuilder.cargs)
            g=GateBuilder.getGate()
            # print(index, end = ' ')
            # if g.cargs :
            #     print(g.carg,end = ' ')
            # print(g.targ, end = ' ')
            # print(g.type().value )
            gates.append(GateBuilder.getGate())

        return gates
        
    @staticmethod
    def _mapping_NISQ(circuit: Circuit, init_mapping: List[int], method: str, num: int, paramter: Dict = {})-> Circuit:
        """
        
        """
       
        mcts_tree = TableBasedMCTS(**paramter)
        mcts_tree.search(logical_circuit = circuit, init_mapping = init_mapping, coupling_graph = circuit.topology)
        return mcts_tree.physical_circuit


        
