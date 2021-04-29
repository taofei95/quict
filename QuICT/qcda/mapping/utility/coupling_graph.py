from __future__ import annotations
import copy
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Callable, Optional, Iterable, Union,Set
from enum import Enum
from collections import deque

import torch 
import torch.nn as nn


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from networkx.algorithms.shortest_paths.dense import reconstruct_path
from networkx.algorithms.shortest_paths.dense import floyd_warshall_predecessor_and_distance

from sklearn.manifold import MDS

from QuICT.tools.interface import *
from QuICT.core.circuit import *
from QuICT.core.gate import *
from QuICT.core.gate.gate import *
from QuICT.core.exception import *
from QuICT.core.layout import *




class CouplingGraph(object):
    def __init__(self, coupling_graph: Union[Layout, List[Tuple[int,int]], None] = None):
        if coupling_graph is not None:
            if isinstance(coupling_graph, Layout):
                self._transform_from_layout(coupling_graph = coupling_graph)
            elif isinstance(coupling_graph, list):
                self._transform_from_list(coupling_graph = coupling_graph)
            else:
                raise Exception("The graph type is not supported")
        else:
            self._coupling_graph = nx.Graph()
        self._size = len(self._coupling_graph.nodes)
        
        self._cal_shortest_path()
        self._generate_matrix_representation()
        self._generate_edge_label()
        self._generate_node_feature()

    @property
    def size(self)->int:
        """
        The number of vertex (physical qubits) in the coupling graph
        """
        return self._size  

    @property
    def num_of_edge(self)->int:
        """
        The number of edges in the coupling graph
        """
        return self._num_of_edge
    @property
    def edges(self)->List[Tuple[int,int]]:
        """
        The list of the edges of the coupling graph
        """
        return self._edges
    @property
    def coupling_graph(self)->nx.Graph:
        """
        """
        return self._coupling_graph
    
    @property
    def node_feature(self)->np.ndarray:
        """
        """
        return self._node_feature
    @property
    def label_matrix(self)->np.ndarray:
        """
        """
        return self._label_matrix
    @property
    def adj_matrix(self)->np.ndarray:
        """
        The adjacent matrix of the coupling graph
        """
        return self._adj_matrix
    @property
    def distance_matrix(self)->np.ndarray:
        """
        The distance matrix of the coupling graph
        """
        return self._distance_matrix
    def get_swap_gate(self, idx: int)-> SwapGate:
        """
        """
        GateBuilder.setGateType(GATE_ID['Swap']) 
        GateBuilder.setTargs([self._edges[idx][0],self._edges[idx][1]])
        return GateBuilder.getGate()

    def get_path(self, source: int, target: int): 
        path = reconstruct_path(source, target, self._predecesors)
        return path

    def edge_label(self, edge: SwapGate)-> int:
        """
        The label of the edge in the graph 
        """
        return self._label_matrix[edge.targs[0],edge.targs[1]]

    def _generate_node_feature(self):
        """
        Generate the feature vector for each node of the graph
        """
        # mds = MDS(dissimilarity='precomputed', n_components=12)
        # mds.fit(self._distance_matrix)
        # embedding = mds.embedding_
        # embedding = (embedding - embedding.min()) / (embedding.max() - embedding.min())
        # self._node_feature = np.zeros((self._size + 1, 12), dtype = np.float32)
        #self._node_feature[1:, 0 : self._size] = self._adj_matrix 
        # self._node_feature[1:, 0 : ] = embedding
        self._node_feature = np.zeros((self._size + 1, self._size), dtype = np.float32)
        self._node_feature[1:,:] = np.identity(self._size)
        
    def _generate_matrix_representation(self):
        """
        Transform the coupling graph to a adjacent matrix representation 
        """
        self._adj_matrix = np.zeros((self._size, self._size), dtype = np.int32)
        self._distance_matrix = np.zeros((self._size, self._size), dtype = np.int32)
        for i in range(self._size):
            for j in range(self._size):
                self._adj_matrix[i,j] = 1 if self.is_adjacent(i,j) else 0
                self._distance_matrix[i,j] = self.distance(i,j)
   
    def _generate_edge_label(self):
        """
        Label the edge in the graph
        """

        
        self._label_matrix = np.zeros((self._size, self._size), dtype = np.int32) - 1 
        self._edges = []
        index = 0
        for i in range(self.size):
            for j in range(i+1, self.size):
                if self._adj_matrix[i,j] == 1:
                    self._label_matrix[i,j] = index
                    self._edges.append((i,j))
                    index += 1
        self._num_of_edge = index
        for i in range(self.size):
            for j in range(0, i):
                self._label_matrix[i,j] = self._label_matrix[j,i]
        
        # for l in self._label_matrix:
        #     print(list(l))
    
    def is_adjacent(self, vertex_i: int, vertex_j: int)-> bool:
        """
        Indicate wthether the two vertices are adjacent on the coupling graph
        """
        if (vertex_i, vertex_j) in self._coupling_graph.edges:
            return True
        else:
            return False
     

    def get_adjacent_vertex(self, vertex: int)->Iterable[int]:
        """
        Return the adjacent vertices of  the given vertex
        """
        return self._coupling_graph.neighbors(vertex)

    def distance(self, vertex_i: int, vertex_j: int)->int:
        """
        The distance of two vertex on the coupling graph of the physical devices
        """
        return self._shortest_paths[vertex_i][vertex_j]


    def _transform_from_list(self,  coupling_graph : List[Tuple[int,int]]):
        """
        Construct the coupling graph from the list of tuples
        """
        res_graph = nx.Graph(coupling_graph)
        self._coupling_graph = res_graph

    def _transform_from_layout(self,  coupling_graph: Layout):
        """
        Construct the coupling graph from the layout class
        """
        edge_list = [ (edge.u, edge.v)  for edge in coupling_graph.edge_list ] 
        res_graph = nx.Graph(edge_list)
        self._coupling_graph = res_graph

    def _cal_shortest_path(self):
        """
        Calculate the shortest path between every two vertices on the graph
        """
        self._predecesors, self._shortest_paths = floyd_warshall_predecessor_and_distance(G = self._coupling_graph)
  
    def draw(self):
        """
        """ 
      
        nx.draw(G = self._coupling_graph)
        plt.savefig("coupling_graph.png")
        plt.close()

ibmq20_topology = [(0,1),(0,5),
            (1,2),(1,6),(1,7),
            (2,3),(2,6),(2,7),
            (3,4),(3,8),(3,9),
            (4,8),(4,9),
            (5,6),(5,10),(5,11),
            (6,7),(6,10),(6,11),
            (7,8),(7,12),(7,13),
            (8,9),(8,12),(8,13),
            (9,14),
            (10,11),(10,15),
            (11,12),(11,16),(11,17),
            (12,13),(12,16),(12,17),
            (13,14),(13,18),(13,19),
            (14,18),(14,19),
            (15,16),
            (16,17),
            (17,18),
            (18,19)]

IBMQ20CouplingGraph = CouplingGraph(ibmq20_topology)

def get_coupling_graph(graph_name: str = None)->CouplingGraph:
    if graph_name == "ibmq20":
        return IBMQ20CouplingGraph
    else:
        raise Exception("No such graph!")
    


