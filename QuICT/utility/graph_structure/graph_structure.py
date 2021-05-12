import importlib.util
import os

cur_dir_files = os.listdir('.')

mod_name = "graph_structure"
mod_path = "graph_structure"

for file in cur_dir_files:
    print(file)
    if file.startswith(mod_path):
        mod_path = file

spec = importlib.util.spec_from_file_location(mod_name, mod_path)
graph_structure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graph_structure)


class GraphStructure:
    def __init__(self, vertex_label: type = int):
        if vertex_label is int:
            self._instance = graph_structure.directed_graph_vertex_label_int()
        elif vertex_label is str:
            self._instance = graph_structure.directed_graph_vertex_label_str()
        else:
            raise NotImplementedError("Only support int/str vertex label for now")

    # print_info
    def print_info(self):
        self._instance.print_info()

    # add_edge
    def add_edge(self, from_, to_, data_):
        self._instance.add_edge(from_, to_, data_)

    # add_vertex
    def add_vertex(self, v):
        self._instance.add_vertex(v)

    # edge_cnt
    def edge_cnt(self) -> int:
        return self._instance.edge_cnt()

    # edges_from
    def edges_from(self, v):
        return self._instance.edges_from(v)

    # edges_to
    def edges_to(self, v):
        return self._instance.edges_to(v)

    # out_deg_of
    def out_deg_of(self, v) -> int:
        return self._instance.out_deg_of(v)

    # in_deg_of
    def in_deg_of(self, v) -> int:
        return self._instance.in_deg_of(v)

    # edges
    def edges(self):
        return self._instance.edges()

    # dijkstra
    def dijkstra(self, source, inf):
        return self._instance.dijkstra(source, inf)
