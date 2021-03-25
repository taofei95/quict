from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.pair cimport pair

cdef extern from "mcts_tree.h" namespace "mcts":
    cdef cppclass MCTSTree:
        MCTSTree() 
        MCTSTree(float gamma, float c,int size_of_subcircuits, int num_of_iterations, int num_of_playout, int num_of_qubits, int feature_dim,
                int num_of_edges,  int* coupling_graph, int* distance_matrix, int* edge_label, float* feature_matrix)
       
        void load_data(int num_of_gates, int num_of_logical_qubits, int* circuit, int* dependency_graph)
        void build_search_tree(vector[int]& qubit_mapping, vector[int]& qubit_mask, vector[int]& front_layer)
        vector[int] search_by_step()
        void print_()
