#ifndef UTILITY
#define UTILITY
#include<vector>
#include<memory>
#include<torch/torch.h>
#include<cassert>

namespace mcts{

void print_tensor(torch::Tensor & data);

struct Gate{
    int ctrl;
    int tar;
    Gate(){
        ctrl = 0;
        tar = 0;
    };
    Gate(int c, int t):ctrl(c), tar(t){};
};


class CouplingGraph{
    public:
        CouplingGraph();
      
        CouplingGraph(int num_of_qubits, int feature_dim, int num_of_edges, int * coupling_graph, int * distance_matrix, int * edge_label, float * feature_matrix);
        CouplingGraph& operator=(CouplingGraph && a);
        void print();
        bool is_adjacent(int v, int u);
        int distance(int v, int u);
        int num_of_qubits;
        int num_of_edges;
        int feature_dim;
        
    private:
  
        std::unique_ptr<torch::TensorAccessor<int, 2>> adj_matrix_accessor;
        std::unique_ptr<torch::TensorAccessor<int, 2>> distance_matrix_accessor;
        std::unique_ptr<torch::TensorAccessor<int, 2>> edge_label_accessor;
        std::unique_ptr<torch::TensorAccessor<float, 2>> feature_matrix_accessor;

        torch::Tensor adj_matrix;
        torch::Tensor distance_matrix;
        torch::Tensor edge_label;
        torch::Tensor feature_matrix;
};


class Circuit{
    public:
        Circuit();
        Circuit(int num_of_gates, int * circuit, int * dependency_graph);
        Circuit& operator=(Circuit&& a);
        std::vector<int> get_succeed_gates(int gate);
        std::vector<int> get_gate_qubits(int gate);
        void print();
    private:
        int num_of_gates;
        std::unique_ptr<torch::TensorAccessor<int, 2>> circuit_accessor;
        std::unique_ptr<torch::TensorAccessor<int, 2>> dependency_graph_accessor;
        torch::Tensor circuit;
        torch::Tensor dependency_graph;
};

}
#endif