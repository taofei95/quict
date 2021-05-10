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
