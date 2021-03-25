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
#include <ctpl.h>
#include <omp.h>
#include <torch/torch.h>
#include <utility.h>
#include <mcts_node.h>



namespace mcts{
float simulate_thread(int id, mcts::MCTSNode cur_node, int seed, int size_of_subcircuits, float gamma); 

class MCTSTree{
    public:
        float gamma;
        float c;
        MCTSTree();
        MCTSTree(float gamma, float c, int size_of_subcircuits, int num_of_iterations, int num_of_playout, int num_of_qubits, int feature_dim,
                int num_of_edges, int * coupling_graph, int * distance_matrix, int * edge_label, float * feature_matrix);
        void load_data(int num_of_gates, int num_of_logical_qubits, int * circuit, int * dependency_graph);
        
        void build_search_tree(std::vector<int>& qubit_mapping, std::vector<int>& qubit_mask, std::vector<int>& front_layer);
        
        void search(std::shared_ptr<MCTSNode> root_node);
        void search_thread(std::shared_ptr<MCTSNode> root_node);
        std::shared_ptr<MCTSNode> select(std::shared_ptr<MCTSNode> root_node);
        void expand(std::shared_ptr<MCTSNode> cur_node);
        float rollout(std::shared_ptr<MCTSNode> cur_node);
        float simulate_multithread(std::shared_ptr<MCTSNode> cur_node);
        float simulate(std::shared_ptr<MCTSNode> cur_node);
        //float simulate_thread(MCTSNode& cur_node, int seed);
        void backpropagate(std::shared_ptr<MCTSNode> cur_node, float value);
        
        std::shared_ptr<MCTSNode> decide(std::shared_ptr<MCTSNode> root_node);
        std::shared_ptr<MCTSNode> select_next_child(std::shared_ptr<MCTSNode> cur_node);

        float upper_condfidence_bound(std::shared_ptr<MCTSNode> cur_node);
        float upper_condfidence_bound_with_predictor(std::shared_ptr<MCTSNode> cur_node);

        std::vector<int> search_by_step();
        void run();
        void print_();
        ~MCTSTree();
        
    private:
        std::shared_ptr<mcts::CouplingGraph> coupling_graph;
        std::shared_ptr<mcts::Circuit> circuit;
        std::shared_ptr<mcts::MCTSNode> root_node;
        //ctpl::thread_pool simulation_pool;
        // int* circuit;
        // int* dependency_graph;
        // int* coupling_graph;
        // int* distance_matrix;
        // float* feature_matrix;
        std::shared_ptr<ctpl::thread_pool>  thread_pool;
        int num_of_swaps;
        int size_of_subcircuits;
        int num_of_logical_qubits;
        int num_of_iterations;
        int num_of_playout;
        int num_of_qubits;
        int num_of_gates;
        int feature_dim;
        
};


}
#endif