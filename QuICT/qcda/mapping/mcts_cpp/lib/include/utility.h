#ifndef UTILITY
#define UTILITY
#include<vector>
#include<memory>
#include<cassert>

namespace mcts{
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
        void print();
        bool is_adjacent(int v, int u);
        int distance(int v, int u);
        int swap_gate_label(Gate& swap_gate);
        std::vector<int> shortest_path(int s, int t);

        int num_of_qubits;
        int num_of_edges;
        int feature_dim;
        std::vector<Gate> edges; 
        
    private:
        int* adj_matrix_accessor;
        int* distance_matrix_accessor;
        int* edge_label_accessor;
        float* feature_matrix_accessor;
};


class Circuit{
    public:
        Circuit();
        Circuit(int num_of_gates, int * circuit, int * dependency_graph);
        std::vector<int> get_succeed_gates(int gate);
        std::vector<int> get_gate_qubits(int gate);
        std::vector<int> get_adj_matrix(std::vector<int>& gates, int length);
        std::vector<int> get_qubits_matrix(std::vector<int>& gates, std::vector<int>& qubit_mapping, int length);

        int num_of_gates;
        int* circuit_accessor;
        int* dependency_graph_accessor;
};

}
#endif