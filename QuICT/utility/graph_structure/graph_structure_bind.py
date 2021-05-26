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


class GraphStructure:
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

    # dijkstra
    def dijkstra(self, source, inf):
        pass


def GraphStructureBuilder(vertex_label: type = int) -> GraphStructure:
    if vertex_label is int:
        return graph_structure_mod.directed_graph_vertex_label_int()
    elif vertex_label is str:
        return graph_structure_mod.directed_graph_vertex_label_str()
    else:
        raise NotImplementedError("Only support int/str vertex label for now")
