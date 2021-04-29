#ifndef MCTS_TREE
#define MCTS_TREE
#include <vector>
#include <iostream>
#include <memory>
#include <cmath>
#include <random>
#include <exception>
#include <ctime>
#include <chrono>
#include <omp.h>
#include <torch/torch.h>
#include "utility.h"
#include "mcts_node.h"

#include "object_pool.h"



namespace mcts{

float simulate_thread(mcts::MCTSNode cur_node, int seed, int num_of_sawp_gates, float gamma); 
float simulate_thread_max(mcts::MCTSNode cur_node, int seed, int num_of_sawp_gates, float gamma); 
float simulate_thread_feng(mcts::MCTSNode cur_node, int seed, int size_of_subcircuits, float gamma);
class MCTSTree{
    public:
      
        MCTSTree();
        MCTSTree(int major ,int method,int info, bool extened, bool with_predictor,
                bool is_generate_data, int threshold_size, 
                float gamma, float c, float virtual_loss, 
                int bp_mode, int num_of_process, 
                int size_of_subcircuits,  int num_of_swap_gates, 
                int num_of_iterations, int num_of_playout,
                int num_of_qubits, int feature_dim, int num_of_edges, 
                int * coupling_graph, int * distance_matrix, 
                int * edge_label, float * feature_matrix);

        void load_data(int num_of_gates, int num_of_logical_qubits,
             int * circuit, int * dependency_graph);
        void build_search_tree(std::vector<int>& qubit_mapping, std::vector<int>& qubit_mask, std::vector<int>& front_layer);
        void delete_tree(MCTSNode* root_node);
        bool is_has_majority(MCTSNode* cur_node);

        std::vector<int> run();
        void search_thread(MCTSNode* root_node);
        MCTSNode* select(MCTSNode* root_node);
        void expand(MCTSNode* cur_node);
        float rollout(MCTSNode* cur_node);
        void backpropagate(MCTSNode* cur_node, float value);
        
        MCTSNode* decide(MCTSNode* root_node);
        MCTSNode* select_next_child(MCTSNode* cur_node);
        MCTSNode* create_child_node_by_swap_gate(MCTSNode* cur_node, int i);
        float upper_condfidence_bound(MCTSNode* cur_node);
        float upper_condfidence_bound_with_predictor(MCTSNode* cur_node);  

        void generate_data(MCTSNode* cur_node, int swap_gate_label);
        void fallback(MCTSNode* cur_node);
        std::vector<int> get_added_swap_label_list();

        int* get_adj_list();
        int* get_qubits_list();
        int* get_num_list();
        int* get_swap_label_list();
        float* get_value_list();
        float* get_action_prob_list();

        int get_num_of_samples();
        int major;
        int method;
        int info;

        ~MCTSTree();

        CouplingGraph coupling_graph;
        Circuit circuit;
        MCTSNode* root_node;

        bool is_send_data = false;
        int fallback_count;
        bool with_predictor;
        bool extended;
        float gamma;
        float c;
        int is_generate_data;
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
        int num_of_gates;
        int feature_dim;
        int num_of_process;
        MCTSNodePool mcts_node_pool;

        std::vector<int> swap_label_list;
        
        int num_of_samples;
        std::vector<std::vector<int>> adj_list;
        std::vector<std::vector<int>> qubits_list;
        std::vector<int> num_list;
        std::vector<int> swap_label_list_n;
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