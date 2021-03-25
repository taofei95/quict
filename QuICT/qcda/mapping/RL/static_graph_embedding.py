#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time    :   2020/12/16 20:53:00
# @Author  :   Jia Zhang 
# @Contact :   jialrs.z@gmail.com
# @File    :   static_graph_embedding.py

import numpy as np
import torch


from coupling_graph import CouplingGraph


class StaticGraphEmbedding(object):
    def __init__(self, graph: CouplingGraph, size  = 10, method = "NN"):
        self._graph = graph
        self._size = size
        self._method = method
        self._embedding_vector = np.zeros(shape = (self._graph.size, size), dtype = np.float)
    
    def __getitem__(self, index):
        return self._embedding_vector[index]

    @property
    def embedding_vector(self)->np.ndarray:
        return self._embedding_vector

    def _matrix_decompostion(self):
        pass
    
    def _nn_embedding(self):
        pass



    
