from libcpp cimport bool
from libcpp.vector cimport vector

cdef extern from "data_generator.h" namespace "mcts":
    cdef cppclass DataGenerator:
        DataGenerator(int major, int info, bool with_predictor, bool is_generate_data, int threshold_of_circuit,
                      float gamma, float c, float virtual_loss, int bp_mode,
                      int num_of_process, int size_of_subcircuits, int num_of_swap_gates, int num_of_iterations,
                      int num_of_playout,
                      int num_of_qubits, int feature_dim, int num_of_edges, bool extended,
                      int *coupling_graph, int *distance_matrix, int *edge_label, float *feature_matrix,
                      const char*model_file_path, int device, int num_of_circuit_process, int inference_batch_size,
                      int max_batch_size)

        void load_data(int num_of_gates, int num_of_logical_qubits, int *circuit, int *dependency_graph,
                       vector[int] qubit_mapping, vector[int] qubit_mask, vector[int] front_layer)
        void update_model()
        void run()
        void clear()
        int get_num_of_samples()
        int*get_adj_list(vector[int]& samples)
        int*get_qubits_list(vector[int]& samples)
        int*get_num_list(vector[int]& samples)
        int*get_swap_label_list(vector[int]& samples)
        float*get_value_list(vector[int]& samples)
        float*get_action_prob_list(vector[int]& samples)
