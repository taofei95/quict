#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/8/22 2:39 下午
# @Author  : Han Yu
# @File    : _mapping.py
import copy

from typing import List, Dict,Tuple,Union
from abc import abstractmethod, abstractclassmethod, abstractstaticmethod

from QuICT.core.circuit import * 
from QuICT.core.exception import *
from QuICT.core.gate import *
from QuICT.core.layout import *
from .utility import *
from .mcts import *


class Mapping(object):  
    @classmethod
    def run(cls, circuit: Circuit, layout: Layout, init_mapping: List[int] = None, init_mapping_method: str = "naive", 
            inplace: bool = False, **parameter)-> Circuit:
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
            init_mapping_method: The method used to dertermine the initial mapping.
                "naive": Using identity mapping, i.e., [0,1,...,n] as the initial mapping.
                "anneal": Using simmulated annealing method[1] to generate the initial mapping.  
            inplace: Indicate wether the algorithm returns a new circuit or modifies the original circuit in place. 
            parameter: The parameters that might  be used in the mapping method.
        Return:  
            the hardware-compliant circuit after mapping.


        [1]Zhou, X., Li, S., & Feng, Y. (2020). Quantum Circuit Transformation Based on Simulated Annealing and Heuristic Search. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 39, 4683-4694.
        """
        circuit.const_lock = True
        num = layout.qubit_number
        gates = cls._run(circuit = circuit, layout = layout, init_mapping = init_mapping, 
                    init_mapping_method = init_mapping_method, **parameter)

        circuit.const_lock = False
        circuit.qubits
        if inplace:
            circuit.set_exec_gates(gates)
        else:
            new_circuit = Circuit(num)
            new_circuit.set_exec_gates(gates)
            return new_circuit
        
    @staticmethod
    def _run(circuit: Circuit, init_mapping: List[int], layout: Layout, init_mapping_method: str, **parameter)-> List[BasicGate]:
        """Use Monte Carlo tree search algorithm to solve the qubit mapping problem
        """
        circuit_dag = DAG(circuit = circuit, mode = Mode.TWO_QUBIT_CIRCUIT)
        coupling_graph = CouplingGraph(coupling_graph=layout)
       
        if init_mapping is None:
            num_of_qubits = coupling_graph.size
            if init_mapping_method == "anneal":
                cost_f = Cost(circuit=circuit_dag, coupling_graph = coupling_graph)
                init_mapping  = np.random.permutation(num_of_qubits)
                _, best_mapping = simulated_annealing(init_mapping = init_mapping, cost = cost_f, method = "nnc",
                            param = {"T_max": 100, "T_min": 1, "alpha": 0.99, "iterations": 1000})
                init_mapping = list(best_mapping)
            elif init_mapping_method == "naive":
                init_mapping = [i for i in range(num_of_qubits)]
            else:
                raise Exception("No such initial mapping method")

        if not isinstance(init_mapping, list):
            raise Exception("Layout should be a list of integers")
        mcts_tree = MCTS(coupling_graph = coupling_graph, info = 0)
        mcts_tree.search(logical_circuit = circuit, init_mapping = init_mapping)
        return mcts_tree.physical_circuit


        
