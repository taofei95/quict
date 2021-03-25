#ifndef MCTS_NODE
#define MCTS_NODE
#include <vector>
#include <memory>
#include <algorithm>
#include <utility.h>
#include <torch/torch.h>
namespace mcts{

class MCTSNode{
    public:
        MCTSNode();
        MCTSNode(std::shared_ptr<mcts::CouplingGraph>& coupling_graph, 
                 std::shared_ptr<mcts::Circuit>& circuit,
                 std::vector<int> &qubit_mapping,
                 std::vector<int> &qubit_mask,
                 std::vector<int> &front_layer,
                 mcts::Gate swap_gate,
                 std::shared_ptr<MCTSNode>&& parent,
                 float prob);

        float value;
        float reward;
        float prob;
        int visit_count;
        mcts::Gate swap_gate; 
        std::shared_ptr<MCTSNode> parent;

        std::vector<int> front_layer;
        std::vector<int> qubit_mapping;
        std::vector<int> inverse_qubit_mapping;
        std::vector<int> qubit_mask;
        
        std::vector<std::shared_ptr<MCTSNode>> children;
        std::vector<Gate> candidate_swap_list;
        std::vector<float> probs_of_children;
       

        int update_by_swap_gate(mcts::Gate& swap_gate);
        void add_child_node_by_swap_gate(mcts::Gate& swap_gate, float prob);
        bool is_leaf_node();
        bool is_terminal_node();
        void update_candidate_swap_list();
        std::vector<int> get_invloved_qubits();
        int update_front_layer();
        bool is_gate_executable(int index);
        bool is_gate_free(int index);
        int nearest_neighbour_cost(std::vector<int>& qubit_mapping);
        int swap_cost(mcts::Gate& swap_gate);

        //~MCTSNode();
    private:
        std::shared_ptr<mcts::CouplingGraph> coupling_graph;
        std::shared_ptr<mcts::Circuit> circuit;

};

}

#endif