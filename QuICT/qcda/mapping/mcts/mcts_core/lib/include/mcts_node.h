#ifndef MCTS_NODE
#define MCTS_NODE
#include <utility.h>

#include <algorithm>
#include <memory>
#include <queue>
#include <vector>
namespace mcts {

class MCTSNode {
 public:
  MCTSNode();
  MCTSNode(CouplingGraph* coupling_graph, Circuit* circuit,
           std::vector<int>& qubit_mapping, std::vector<int>& qubit_mask,
           std::vector<int>& front_layer, mcts::Gate swap_gate,
           MCTSNode* parent, float prob);
  MCTSNode(MCTSNode& node);
  void clear();
  MCTSNode& operator=(MCTSNode&& node);

  float value;
  float w;
  float reward;
  float prob;
  int visit_count;

  int num_of_children;

  int num_of_gates;
  mcts::Gate swap_gate;
  MCTSNode* parent;
  MCTSNode* brother;
  MCTSNode* child;

  std::vector<int> front_layer;
  std::vector<int> qubit_mapping;
  std::vector<int> inverse_qubit_mapping;
  std::vector<int> qubit_mask;

  std::vector<Gate> candidate_swap_list;
  std::vector<float> probs_of_children;
  CouplingGraph* coupling_graph;
  Circuit* circuit;

  int update_by_swap_gate(mcts::Gate& swap_gate);
  bool is_leaf_node();
  bool is_terminal_node();
  void update_candidate_swap_list();
  void update_candidate_swap_list_extended();
  std::vector<int> get_subcircuit(int num_of_circuit);
  std::vector<int> get_invloved_qubits();
  int update_front_layer();
  bool is_gate_executable(int index);
  bool is_gate_free(int index);
  bool is_gate_free(int gate, std::vector<int>& qubit_mapping);
};

}  // namespace mcts

#endif