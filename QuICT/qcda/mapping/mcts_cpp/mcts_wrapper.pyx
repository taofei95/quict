# distutils: language = c++
# distutils: extra_compile_args = -std=c++14 -I /home/shoulifu/QuICT/QuICT/qcda/mapping/mcts_cpp/lib/include/ -I /home/shoulifu/libtorch/include/torch/csrc/api/include -I /home/shoulifu/libtorch/include
# distutils: extra_link_args = -L/home/shoulifu/QuICT/QuICT/qcda/mapping/mcts_cpp/lib/build/
# cython: language_level = 3
# cython: boundscheck = False
# cython: wraparound = False 

cimport numpy as np

from cython.operator cimport dereference as deref
from libcpp.vector cimport vector
from libcpp.algorithm cimport sort
from libcpp.memory cimport shared_ptr, make_shared
from mcts_cpp cimport MCTSTree



cdef class MCTSTreeWrapper:
    cdef:
        MCTSTree* mcts_cpp
        int[:, ::1] circuit, dependency_graph, coupling_graph, distance_matrix, edge_label
        float[:, ::1] feature_matrix
        int num_of_iterations, num_of_logical_qubits, num_of_playout
        float gamma, c

    def __cinit__(self, float gamma, float c, int size_of_subcircuits, int num_of_iterations, int num_of_playout, int num_of_edges,  
                np.ndarray[np.int32_t, ndim = 2] coupling_graph, np.ndarray[np.int32_t, ndim = 2] distance_matrix, np.ndarray[np.int32_t, ndim = 2] edge_label, np.ndarray[np.float32_t, ndim = 2] feature_matrix):
        self.coupling_graph = coupling_graph
        self.distance_matrix = distance_matrix
        self.feature_matrix = feature_matrix
        self.edge_label = edge_label
        #print((&self.feature_matrix[0,0]))
        #self.mcts_cpp
        self.mcts_cpp = new MCTSTree(gamma, c, size_of_subcircuits, num_of_iterations, num_of_playout, <int>self.coupling_graph.shape[0], <int>self.feature_matrix.shape[1], num_of_edges, &(self.coupling_graph[0,0]), &(self.distance_matrix[0,0]), &(self.edge_label[0][0]), &(self.feature_matrix[0,0]))


    def load_data(self,int num_of_logical_qubits, np.ndarray[np.int32_t, ndim = 2] circuit, np.ndarray[np.int32_t, ndim =2] dependency_graph, list qubit_mapping, list qubit_mask, list front_layer):   
        self.circuit = circuit
        self.dependency_graph = dependency_graph
        deref(self.mcts_cpp).load_data(self.circuit.shape[0], num_of_logical_qubits, &self.circuit[0][0], &self.dependency_graph[0][0])
        deref(self.mcts_cpp).build_search_tree(qubit_mapping, qubit_mask, front_layer)
        # self.mcts_cpp[0].print_()

    def search(self):
        return deref(self.mcts_cpp).search_by_step()
    
    
