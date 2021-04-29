#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/8/22 2:39 下午
# @Author  : Han Yu
# @File    : _mapping.py
import copy

from typing import List, Dict,Tuple,Union
from abc import abstractmethod, abstractclassmethod, abstractstaticmethod

from networkx.classes import graph
from QuICT.core.circuit import * 
from QuICT.core.exception import *
from QuICT.core.gate import *
from QuICT.core.layout import *
from .utility.dag import DAG
from .utility.coupling_graph import CouplingGraph
from .utility.utility import Mode
from .mcts.mcts import *
from .utility.init_mapping import simulated_annealing, Cost


class Mapping(object):  
    @classmethod
    def run(cls, circuit: Circuit, layout: Layout, init_mapping: List[int] = None, inplace: bool = False, **parameter)-> Circuit:
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
        num = circuit.circuit_width()
        gates = cls._run(circuit = circuit, layout = layout, init_mapping = init_mapping , **parameter)
        
        circuit.const_lock = False
        circuit.qubits
        if inplace:
            circuit.set_flush_gates(gates)
        else:
            new_circuit = Circuit(num)
            new_circuit.extend(gates)
            return new_circuit
        
    @staticmethod
    def _run(circuit: Circuit, init_mapping: List[int], layout: Layout, **parameter)-> Circuit:
        """Use Monte Carlo tree search algorithm to solve the qubit mapping problem
        """
        circuit_dag = DAG(circuit = circuit, mode = Mode)
        coupling_graph = CouplingGraph(coupling_graph=layout)
        if init_mapping is None:
            num_of_qubits = circuit.circuit_width()
            cost_f = Cost(circuit=circuit_dag, coupling_graph = coupling_graph)
            init_mapping  = np.random.permutation(num_of_qubits)
            _, best_mapping = simulated_annealing(init_mapping = init_mapping, cost = cost_f, method = "nnc",
                         param = {"T_max": 100, "T_min": 1, "alpha": 0.98, "iterations": 1000})
            init_mapping = best_mapping

        if isinstance(init_mapping, list):
            raise Exception("Layout should be a list of integers")
        mcts_tree = MCTS(coupling_graph = coupling_graph)
        mcts_tree.search(logical_circuit = circuit, init_mapping = init_mapping)
        return mcts_tree.physical_circuit


        
