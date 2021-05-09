from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.pair cimport pair

cdef extern from "mcts_tree.h" namespace "mcts":
    cdef cppclass MCTSTree:
        #~MCTSTree()
        MCTSTree(int major, int method, int info, bool extended, bool with_predictor, bool is_generate_data, 
                int threshold_size, float gamma, float c, float virtual_loss, 
                int bp_mode, int num_of_process, int size_of_subcircuits, 
                int num_of_swap_gates, int num_of_iterations, int num_of_playout, int num_of_qubits, int feature_dim,
                int num_of_edges,  int* coupling_graph, int* distance_matrix, int* edge_label, float* feature_matrix)
       
        void load_data(int num_of_gates, int num_of_logical_qubits, int* circuit, int* dependency_graph)
        vector[int] get_added_swap_label_list()
        void build_search_tree(vector[int]& qubit_mapping, vector[int]& qubit_mask, vector[int]& front_layer)
        vector[int] run()
        void print_()
        
        int get_num_of_samples()
        int* get_adj_list()
        int* get_qubits_list()
        int* get_num_list()
        int* get_swap_label_list()
        float* get_value_list()
        float* get_action_prob_list()
        
# cdef extern from "mcts_tree.h" namespace "mcts":
#     cdef cppclass RLMCTSTree(MCTSTree):
#         RLMCTSTree(int major, int info, bool extended, bool with_predictor, bool is_generate_data, 
#                 int threshold_of_circuit, float gamma, float c, 
#                 float virtual_loss, int bp_mode, 
#                 int num_of_process, int size_of_subcircuits, 
#                 int num_of_swap_gates, int num_of_iterations, 
#                 int num_of_playout, 
#                 int num_of_qubits, int feature_dim,  
#                 int num_of_edges, int * coupling_graph, 
#                 int * distance_matrix, int * edge_label, 
#                 float * feature_matrix, const string model_file_path, int device)
        
#         vector[int] run_rl()



    
        

