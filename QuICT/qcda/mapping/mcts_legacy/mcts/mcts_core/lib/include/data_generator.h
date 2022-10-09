#ifndef DATA_GENERATOR
#define DATA_GENERATOR
#include "rl_mcts_tree.h"

namespace mcts{
    class DataGenerator{
        public:
            DataGenerator(int major, int info, bool with_predictor, 
                        bool is_generate_data, int threshold_of_circuit, 
                        float gamma, float c, 
                        float virtual_loss, int bp_mode, 
                        int num_of_process, int size_of_subcircuits, 
                        int num_of_swap_gates, int num_of_iterations, 
                        int num_of_playout, int num_of_qubits, 
                        int feature_dim, int num_of_edges, 
                        bool extended, int * coupling_graph, 
                        int * distance_matrix, int * edge_label, 
                        float * feature_matrix, const char* model_file_path, 
                        int device, int num_of_circuit_process, 
                        int inference_batch_size, int max_batch_size);
            ~DataGenerator();
            void load_data(int num_of_gates, int num_of_logical_qubits, 
                        int * circuit, int * dependency_graph, 
                        std::vector<int> qubit_mapping, std::vector<int> qubit_mask, 
                        std::vector<int> front_layer);
            void run();
            void clear();
            void update_model();

            int* get_adj_list(std::vector<int>& batch_samples);
            int* get_qubits_list(std::vector<int>& batch_samples);
            int* get_num_list(std::vector<int>& batch_samples);
            int* get_swap_label_list(std::vector<int>& batch_samples);
            float* get_value_list(std::vector<int>& batch_samples);
            float* get_action_prob_list(std::vector<int>& batch_samples);
            int get_num_of_samples();
            int major; 
            int info;
            int max_batch_size;
            int num_of_process;
            int num_of_circuit_process;
            bool with_predictor;
            bool extended;
            float gamma;
            float c;
            bool is_generate_data;
            int threshold_size;
            int num_of_executed_gates;
            int num_of_added_swap_gates;
            int bp_mode;
            int virtual_loss;

            int num_of_swap_gates;
            int size_of_subcircuits;
            int num_of_logical_qubits;
            int num_of_iterations;
            int num_of_playout;
            int num_of_qubits;
  
            std::string model_file_path;
            int device;
            int inference_batch_size;
            CouplingGraph coupling_graph;
            Circuit circuit;

            std::thread inferencer;
            SampleQueue sample_queue;
            std::atomic_flag* thread_flag;
            std::atomic_flag* res_flags;
            float* res_values;
            torch::Tensor* res_probs;

            std::atomic_flag pool_lock;
            std::atomic_int num_of_samples;
            int num_of_circuit;
            std::vector<Circuit> circuit_list;

            std::vector<std::vector<int>> qubit_mapping_list;
            std::vector<std::vector<int>> qubit_mask_list; 
            std::vector<std::vector<int>> front_layer_list;

            std::vector<std::vector<int>> adj_list;
            std::vector<std::vector<int>> qubits_list;
            std::vector<int> num_list;
            std::vector<int> swap_label_list;
            std::vector<float> value_list;
            std::vector<std::vector<float>> action_prob_list;


            int* adj_mem;
            int* qubits_mem;
            int* num_mem;
            int* swap_label_mem;
            float* value_mem;
            float* action_prob_mem;  

            
    };
}
#endif