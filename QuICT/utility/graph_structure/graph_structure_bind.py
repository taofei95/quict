import importlib.util
import os
from typing import *

cur_path = os.path.dirname(os.path.abspath(__file__))

# print(cur_path)

mod_name = "graph_structure"
mod_path = "graph_structure"

for file in os.listdir(cur_path):
    print(file)
    if file.startswith(mod_path):
        mod_path = file

spec = importlib.util.spec_from_file_location(mod_name, mod_path)
graph_structure_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graph_structure_mod)


class Vertex:
    def __init__(self, id_, attribute):
        self.id = id_
        self.attribute = attribute

    def __str__(self):
        return f"id = {self.id}, attribute = {self.attribute}"


class Edge:
    def __init__(self, from_: Vertex, to_: Vertex, data_: int):
        self.from_: Vertex = from_
        self.to_: Vertex = to_
        self.data_: int = data_

    def print_info(self):
        pass


class DirectedGraph:
    def __init__(self):
        self._instance = directed_graph_builder(int)
        self._vertex_by_id = {}
        self._index_cnt = 0

    def get_vertex(self, id: int) -> Vertex:
        if id not in self._vertex_by_id:
            raise IndexError(f"No vertex in this graph with ID {id}")
        else:
            return self._vertex_by_id[id]

    # print_info
    def print_info(self):
        self._instance.print_info()

    # add_edge
    def add_edge(self, from_: Vertex, to_: Vertex, data_: int):
        self.add_node(from_)
        self.add_node(to_)
        self._instance.add_edge(from_.id, to_.id, data_)
        return self

    # add_vertex
    def add_node(self, v: Vertex):
        if v.id not in self._vertex_by_id:
            self._vertex_by_id[v.id] = v

    # edge_cnt
    def edge_cnt(self) -> int:
        return self._instance.edge_cnt()

    # edges_from
    def edges_from(self, v: Vertex) -> Iterable[Edge]:
        return self._instance.edges_from(v.id)

    # edges_to
    def edges_to(self, v: Vertex) -> Iterable[Edge]:
        return self._instance.edges_to(v.id)

    # out_deg_of
    def out_degree(self, v: Vertex) -> int:
        return self._instance.out_degree(v.id)

    # in_deg_of
    def in_degree(self, v: Vertex) -> int:
        return self._instance.in_degree(v.id)

    # edges
    def edges(self) -> Iterable[Edge]:
        return self._instance.edges()

    def __str__(self):
        return self._instance.info_str()


def directed_graph_builder(vertex_label: type = int) -> DirectedGraph:
    if vertex_label is int:
        return graph_structure_mod.directed_graph_vertex_label_int()
    elif vertex_label is str:
        return graph_structure_mod.directed_graph_vertex_label_str()
    else:
        raise NotImplementedError("Only int/str are supported!")
