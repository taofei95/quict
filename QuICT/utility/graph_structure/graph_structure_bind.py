import importlib.util
import os
from typing import *

cur_path = os.path.dirname(os.path.abspath(__file__))

print(cur_path)

mod_name = "graph_structure"
mod_path = "graph_structure"

for file in os.listdir(cur_path):
    print(file)
    if file.startswith(mod_path):
        mod_path = file

spec = importlib.util.spec_from_file_location(mod_name, mod_path)
graph_structure_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graph_structure_mod)


class Edge:
    def __init__(self):
        self.from_ = None
        self.to_ = None
        self.data_ = None

    def print_info(self):
        pass


class DirectedGraph:
    # print_info
    def print_info(self):
        pass

    # add_edge
    def add_edge(self, from_, to_, data_):
        pass

    # add_vertex
    def add_vertex(self, v):
        pass

    # edge_cnt
    def edge_cnt(self) -> int:
        pass

    # edges_from
    def edges_from(self, v) -> Iterable[Edge]:
        pass

    # edges_to
    def edges_to(self, v) -> Iterable[Edge]:
        pass

    # out_deg_of
    def out_deg_of(self, v) -> int:
        pass

    # in_deg_of
    def in_deg_of(self, v) -> int:
        pass

    # edges
    def edges(self) -> Iterable[Edge]:
        pass


class DirectedGraphWrapper:
    def __init__(self):
        self._instance = directed_graph_builder(int)
        self.index_to_label = {}
        self.label_to_index = {}
        self._index_cnt = 0

    # print_info
    def print_info(self):
        self._instance.print_info()

    # add_edge
    def add_edge(self, from_, to_, data_):
        self.add_vertex(from_)
        self.add_vertex(to_)
        self._instance.add_edge(self.label_to_index[from_], self.label_to_index[to_], data_)

    # add_vertex
    def add_vertex(self, v):
        if v not in self.label_to_index:
            self._instance.add_vertex(self._index_cnt)
            self.label_to_index[v] = self._index_cnt
            self.index_to_label[self._index_cnt] = v
            self._index_cnt += 1

    # edge_cnt
    def edge_cnt(self) -> int:
        return self._instance.edge_cnt()

    # edges_from
    def edges_from(self, v) -> Iterable[Edge]:
        return self._instance.edges_from(self.label_to_index[v])

    # edges_to
    def edges_to(self, v) -> Iterable[Edge]:
        return self._instance.edges_to(self.label_to_index[v])

    # out_deg_of
    def out_deg_of(self, v) -> int:
        return self._instance.out_deg_of(self.label_to_index[v])

    # in_deg_of
    def in_deg_of(self, v) -> int:
        return self._instance.in_deg_of(self.label_to_index[v])

    # edges
    def edges(self) -> Iterable[Edge]:
        return self._instance.edges()


def directed_graph_builder(vertex_label: type = int) -> DirectedGraph:
    if vertex_label is int:
        return graph_structure_mod.directed_graph_vertex_label_int()
    elif vertex_label is str:
        return graph_structure_mod.directed_graph_vertex_label_str()
    else:
        return DirectedGraphWrapper()
