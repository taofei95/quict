#include <mcts_tree.h>

#include <iostream>

float random_generator(int seed) {
  std::mt19937_64 gen = std::mt19937_64((unsigned int)(time(NULL)) * seed);
  std::uniform_real_distribution<float> dist =
      std::uniform_real_distribution<float>(0.0, 1.0);
  return dist(gen);
}

int choice(std::vector<float>& p, int seed) {
  float r = random_generator(seed), t = 0;
  for (unsigned int i = 0; i < p.size(); i++) {
    t += p[i];
    if (t > r) {
      return i;
    }
  }
  return p.size() - 1;
}

int argmin(std::vector<int>& p, int seed) {
  int m = INT_MAX, t = 0;
  for (unsigned int i = 0; i < p.size(); i++) {
    if (p[i] < m) {
      m = p[i];
      t = i;
    } else if (p[i] == m) {
      float r = random_generator(seed);
      if (r > 0.5) {
        t = i;
      }
    }
  }
  return t;
}

int nearest_neighbour_cost(mcts::MCTSNode& cur_node, std::vector<int>& gates,
                           std::vector<int>& cur_mapping) {
  int res = 0;
  for (auto& g : gates) {
    std::vector<int> qubits = cur_node.circuit->get_gate_qubits(g);
    res += cur_node.coupling_graph->distance(cur_mapping[qubits[0]],
                                             cur_mapping[qubits[1]]);
  }
  return res;
}

int swap_cost(mcts::MCTSNode& cur_node, mcts::Gate& swap_gate) {
  int res = 0;
  std::vector<int> cur_mapping = cur_node.qubit_mapping;
  std::vector<int> p_qubits = {swap_gate.ctrl, swap_gate.tar};
  std::vector<int> l_qubits = {cur_node.inverse_qubit_mapping[swap_gate.ctrl],
                               cur_node.inverse_qubit_mapping[swap_gate.tar]};
  assert(cur_node.coupling_graph->is_adjacent(p_qubits[0], p_qubits[1]));
  cur_mapping[l_qubits[0]] = p_qubits[1];
  cur_mapping[l_qubits[1]] = p_qubits[0];
  res = nearest_neighbour_cost(cur_node, cur_node.front_layer, cur_mapping);
  return res;
}

void normalize(std::vector<float>& x) {
  float sum = 0.0;
  for (auto& p : x) {
    sum += p;
  }
  for (auto& p : x) {
    p = p / sum;
  }
}

float f(float x) {
  if (x < 0) {
    return 0.0;
  } else if (x == 0) {
    return 0.001;
  } else {
    return x;
  }
}

float mcts::simulate_thread(mcts::MCTSNode cur_node, int seed,
                            int num_of_swap_gates, float gamma) {
  float res = 0, weight = 1.0;
  int num_of_executed_gates = 0, num_of_added_swap_gates = 0;
  while (num_of_added_swap_gates < num_of_swap_gates &&
         !cur_node.is_terminal_node()) {
    cur_node.update_candidate_swap_list();
    std::vector<float> heuristic_prob(cur_node.num_of_children, 0.0);
    int nnc_base = nearest_neighbour_cost(cur_node, cur_node.front_layer,
                                          cur_node.qubit_mapping);
    for (int i = 0; i < cur_node.num_of_children; i++) {
      heuristic_prob[i] = f(static_cast<float>(
          nnc_base - swap_cost(cur_node, cur_node.candidate_swap_list[i])));
    }
    normalize(heuristic_prob);
    int idx = choice(heuristic_prob, seed);
    num_of_executed_gates =
        cur_node.update_by_swap_gate(cur_node.candidate_swap_list[idx]);
    num_of_added_swap_gates += 1;
    res += weight * num_of_executed_gates;
    weight *= gamma;
  }

  return res;
}

float mcts::simulate_thread_max(mcts::MCTSNode cur_node, int seed,
                                int num_of_swap_gates, float gamma) {
  float res = 0, weight = 1.0;
  int num_of_executed_gates = 0, num_of_added_swap_gates = 0;
  while (num_of_added_swap_gates < num_of_swap_gates &&
         !cur_node.is_terminal_node()) {
    cur_node.update_candidate_swap_list();
    std::vector<int> heuristic_prob(cur_node.num_of_children, 0.0);
    for (int i = 0; i < cur_node.num_of_children; i++) {
      heuristic_prob[i] = swap_cost(cur_node, cur_node.candidate_swap_list[i]);
    }
    int idx = argmin(heuristic_prob, seed);
    num_of_executed_gates =
        cur_node.update_by_swap_gate(cur_node.candidate_swap_list[idx]);
    num_of_added_swap_gates += 1;
    res += weight * num_of_executed_gates;
    weight *= gamma;
  }
  return res;
}

float mcts::simulate_thread_feng(mcts::MCTSNode cur_node, int seed,
                                 int size_of_subcircuits, float gamma) {
  float res = 0, weight = 1.0;
  float num_of_executed_gates = 0, num_of_added_swap_gates = 0;
  while (num_of_executed_gates < size_of_subcircuits &&
         !cur_node.is_terminal_node()) {
    cur_node.update_candidate_swap_list();
    std::vector<float> heuristic_prob(cur_node.num_of_children, 0.0);
    int nnc_base = nearest_neighbour_cost(cur_node, cur_node.front_layer,
                                          cur_node.qubit_mapping);
    for (int i = 0; i < cur_node.num_of_children; i++) {
      heuristic_prob[i] = f(static_cast<float>(
          nnc_base - swap_cost(cur_node, cur_node.candidate_swap_list[i])));
    }
    normalize(heuristic_prob);
    int idx = choice(heuristic_prob, seed);
    num_of_executed_gates +=
        cur_node.update_by_swap_gate(cur_node.candidate_swap_list[idx]);
    num_of_added_swap_gates += 1;
  }
  if (num_of_executed_gates > size_of_subcircuits) {
    res = std::pow(gamma, num_of_added_swap_gates / 2) *
          (float)(size_of_subcircuits);
  } else {
    res = std::pow(gamma, num_of_added_swap_gates / 2) *
          (float)(num_of_executed_gates);
  }
  return res;
}

mcts::MCTSTree::MCTSTree() {
  this->gamma = 0.7;
  this->c = 20;
  this->num_of_iterations = 40;
  this->num_of_playout = 500;
  this->size_of_subcircuits = 30;
}

mcts::MCTSTree::MCTSTree(int major, int method, int info, bool extended,
                         bool with_predictor, bool is_generate_data,
                         int threshold_size, float gamma, float c,
                         float virtual_loss, int bp_mode, int num_of_process,
                         int size_of_subcircuits, int num_of_swap_gates,
                         int num_of_iterations, int num_of_playout,
                         int num_of_qubits, int feature_dim, int num_of_edges,
                         int* coupling_graph, int* distance_matrix,
                         int* edge_label, float* feature_matrix) {
  this->major = major;
  this->method = method;
  this->info = info;
  this->bp_mode = bp_mode;
  this->extended = extended;
  this->with_predictor = with_predictor;
  this->num_of_samples = 0;
  this->fallback_count = 0;
  this->num_of_added_swap_gates = 0;
  this->num_of_executed_gates = 0;
  this->with_predictor = with_predictor;
  this->is_generate_data = is_generate_data;
  this->threshold_size = threshold_size;
  this->virtual_loss = virtual_loss;
  this->num_of_process = num_of_process;
  this->gamma = gamma;
  this->c = c;
  this->num_of_iterations = num_of_iterations;
  this->num_of_playout = num_of_playout;
  this->num_of_qubits = num_of_qubits;
  this->feature_dim = feature_dim;
  this->size_of_subcircuits = size_of_subcircuits;
  this->num_of_swap_gates = num_of_swap_gates;

  this->coupling_graph = std::move(mcts::CouplingGraph(
      num_of_qubits, feature_dim, num_of_edges, coupling_graph, distance_matrix,
      edge_label, feature_matrix));
  long capacity = static_cast<long>(this->num_of_iterations) *
                  static_cast<long>(num_of_edges) * 500;
  this->mcts_node_pool.resize(capacity);
}

mcts::MCTSTree::~MCTSTree() {
  if (this->is_send_data) {
    delete[] adj_mem, qubits_mem, num_mem, swap_label_mem, value_mem,
        action_prob_mem;
  }
}

void mcts::MCTSTree::delete_tree(mcts::MCTSNode* cur_node) {
  if (cur_node != nullptr) {
    mcts::MCTSNode* child_node = cur_node->child;
    while (child_node != nullptr) {
      if (child_node->parent != nullptr) {
        this->delete_tree(child_node);
      }
      child_node = child_node->brother;
    }
    this->mcts_node_pool.destroy(cur_node);
  }
}

bool mcts::MCTSTree::is_has_majority(mcts::MCTSNode* cur_node) {
  if (this->major > 1) {
    mcts::MCTSNode* child_node = cur_node->child;
    int minimum = this->num_of_iterations, maximum = -1, sum = 0;
    while (child_node != nullptr) {
      sum += child_node->visit_count;
      minimum = std::min(minimum, child_node->visit_count);
      maximum = std::max(maximum, child_node->visit_count);
      child_node = child_node->brother;
    }
    return minimum > 1 && this->major * maximum > (this->major - 1) * sum;
  } else if (this->major == 1) {
    return true;
  } else {
    return false;
  }
}

void mcts::MCTSTree::load_data(int num_of_gates, int num_of_logical_qubits,
                               int* circuit, int* dependency_graph) {
  this->num_of_gates = num_of_gates;
  this->num_of_logical_qubits = num_of_logical_qubits;
  this->circuit =
      std::move(mcts::Circuit(num_of_gates, circuit, dependency_graph));
  this->adj_list.clear();
  this->qubits_list.clear();
  this->swap_label_list_n.clear();
  this->swap_label_list.clear();
  this->value_list.clear();
  this->action_prob_list.clear();
  this->num_list.clear();
  this->num_of_added_swap_gates = 0;
  this->num_of_executed_gates = 0;
  this->num_of_samples = 0;
  this->fallback_count = 0;
  this->mcts_node_pool.clear();
}

void mcts::MCTSTree::build_search_tree(std::vector<int>& qubit_mapping,
                                       std::vector<int>& qubit_mask,
                                       std::vector<int>& front_layer) {
  this->root_node = this->mcts_node_pool.create();
  *(this->root_node) = std::move(
      mcts::MCTSNode(&(this->coupling_graph), &(this->circuit), qubit_mapping,
                     qubit_mask, front_layer, mcts::Gate(), nullptr, 0.0));
  if (this->extended)
    root_node->update_candidate_swap_list_extended();
  else
    root_node->update_candidate_swap_list();
  if (!root_node->is_terminal_node()) this->expand(root_node);
}

std::vector<int> mcts::MCTSTree::run() {
  mcts::MCTSNode *cur_node = this->root_node, *pre_node = nullptr;
  this->num_of_executed_gates += cur_node->reward;
  while (!cur_node->is_terminal_node()) {
    auto start = std::chrono::system_clock::now();
    if (!is_has_majority(cur_node)) {
      if (this->num_of_process > 1) {
#pragma omp parallel for num_threads(this->num_of_process)
        for (int i = 0; i < this->num_of_iterations; i++) {
          this->search_thread(cur_node);
        }
      } else {
        for (int i = 0; i < this->num_of_iterations; i++) {
          this->search_thread(cur_node);
          // std::cout<<i<<std::endl;
        }
      }
    }
    pre_node = cur_node;
    cur_node = this->decide(cur_node);
    cur_node->parent = nullptr;
    if (cur_node->reward == 0) {
      this->fallback_count += 1;
    } else {
      this->fallback_count = 0;
    }
    this->num_of_added_swap_gates += 1;
    int best_swap_label =
        this->coupling_graph.swap_gate_label(cur_node->swap_gate);
    this->swap_label_list.push_back(best_swap_label);
    this->num_of_executed_gates += cur_node->reward;
    if (this->is_generate_data) {
      this->generate_data(pre_node, best_swap_label);
    }
    this->delete_tree(pre_node);
    if (this->fallback_count > 15) {
      this->fallback(cur_node);
    }
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> cost_time = end - start;
    if (this->info == 1) {
      std::cout << "------------------------------------------" << std::endl;
      std::cout << " cost time: " << cost_time.count() << std::endl;
      std::cout << " swap gate:   " << this->num_of_added_swap_gates
                << " executed gate:   " << this->num_of_executed_gates
                << " num of remaining gates:  " << cur_node->num_of_gates
                << std::endl;
      std::cout << "------------------------------------------" << std::endl;

      std::cout << "memory used: " << this->mcts_node_pool.getSize()
                << std::endl;
    }
  }

  std::vector<int> res{this->num_of_executed_gates,
                       this->num_of_added_swap_gates};
  return res;
}

void mcts::MCTSTree::search_thread(mcts::MCTSNode* root_node) {
  mcts::MCTSNode* cur_node = this->select(root_node);
  if (!cur_node->is_terminal_node()) {
    this->expand(cur_node);
  }
  float value = this->rollout(cur_node);
  this->backpropagate(cur_node, value);
}

mcts::MCTSNode* mcts::MCTSTree::select(mcts::MCTSNode* root_node) {
  mcts::MCTSNode* cur_node = root_node;
  cur_node->visit_count += (this->virtual_loss + 1);
  while (!cur_node->is_leaf_node()) {
    cur_node = this->select_next_child(cur_node);
    cur_node->visit_count += (this->virtual_loss + 1);
  }
  return cur_node;
}

mcts::MCTSNode* mcts::MCTSTree::select_next_child(mcts::MCTSNode* cur_node) {
  float score = -1;
  mcts::MCTSNode *res_node, *child = cur_node->child;
  while (child != nullptr) {
    float ucb;
    if (with_predictor) {
      ucb = this->upper_condfidence_bound_with_predictor(child);
    } else {
      ucb = this->upper_condfidence_bound(child);
    }
    if (ucb > score) {
      res_node = child;
      score = ucb;
    }
    child = child->brother;
  }
  return res_node;
}

float mcts::MCTSTree::upper_condfidence_bound(mcts::MCTSNode* cur_node) {
  return this->gamma * cur_node->value + cur_node->reward +
         this->c * sqrt(log((float)(cur_node->parent->visit_count)) /
                        (cur_node->visit_count + 0.001));
}

float mcts::MCTSTree::upper_condfidence_bound_with_predictor(
    mcts::MCTSNode* cur_node) {
  return this->gamma * cur_node->value + cur_node->reward +
         this->c * cur_node->prob *
             sqrt((float)(cur_node->parent->visit_count) /
                  (1.0 + cur_node->visit_count));
}

void mcts::MCTSTree::expand(mcts::MCTSNode* cur_node) {
  assert(cur_node->candidate_swap_list.size() ==
         cur_node->probs_of_children.size());
  int num_of_swap_gates = cur_node->candidate_swap_list.size();
  mcts::MCTSNode* child = this->create_child_node_by_swap_gate(cur_node, 0);
  mcts::MCTSNode* first_child = child;
  for (int i = 1; i < num_of_swap_gates; i++) {
    child->brother = create_child_node_by_swap_gate(cur_node, i);
    child = child->brother;
    // std::cout<<i<<std::endl;
  }
  cur_node->child = first_child;
}

void mcts::MCTSTree::backpropagate(mcts::MCTSNode* cur_node, float value) {
  float bp_value = value;
  if (this->bp_mode == 0) {
    cur_node->visit_count -= this->virtual_loss;
    bp_value = cur_node->reward + this->gamma * bp_value;
    cur_node = cur_node->parent;
    while (cur_node != nullptr) {
      cur_node->visit_count -= this->virtual_loss;
      if (bp_value > cur_node->value) {
        cur_node->value = bp_value;
      }
      bp_value = cur_node->reward + this->gamma * bp_value;
      cur_node = cur_node->parent;
    }
  } else if (this->bp_mode == 1) {
    cur_node->visit_count -= this->virtual_loss;
    bp_value = cur_node->reward + this->gamma * bp_value;
    cur_node = cur_node->parent;
    while (cur_node != nullptr) {
      cur_node->visit_count -= this->virtual_loss;
      cur_node->w += bp_value;
      cur_node->value = cur_node->value / cur_node->visit_count;

      bp_value = cur_node->reward + this->gamma * bp_value;
      cur_node = cur_node->parent;
    }
  } else {
    assert(this->bp_mode < 2);
  }
}

float mcts::MCTSTree::rollout(MCTSNode* cur_node) {
  std::vector<float> res(this->num_of_playout);
#pragma omp parallel for
  for (int i = 0; i < this->num_of_playout; i++) {
    if (this->method == 0)
      res[i] =
          simulate_thread(*cur_node, i, this->num_of_swap_gates, this->gamma);
    else if (this->method == 1)
      res[i] = simulate_thread_feng(*cur_node, i, this->size_of_subcircuits,
                                    this->gamma);
    else if (this->method == 2)
      res[i] = simulate_thread_max(*cur_node, i, this->num_of_swap_gates,
                                   this->gamma);
    else
      assert(this->method <= 2);
  }

  float maximum = -1;
  for (int i = 0; i < this->num_of_playout; i++) {
    if (res[i] > maximum) {
      maximum = res[i];
    }
  }
  cur_node->value = maximum;
  return maximum;
}

mcts::MCTSNode* mcts::MCTSTree::create_child_node_by_swap_gate(
    mcts::MCTSNode* cur_node, int i) {
  int a = cur_node->candidate_swap_list.size();
  // std::cout<<a<<std::endl;
  assert(i < cur_node->candidate_swap_list.size());
  mcts::Gate swap_gate = cur_node->candidate_swap_list[i];
  std::vector<int> p_qubits = {swap_gate.ctrl, swap_gate.tar};
  std::vector<int> l_qubits = {cur_node->inverse_qubit_mapping[swap_gate.ctrl],
                               cur_node->inverse_qubit_mapping[swap_gate.tar]};

  std::vector<int> qubit_mapping = cur_node->qubit_mapping;
  std::vector<int> qubit_mask = cur_node->qubit_mask;

  assert(cur_node->coupling_graph->is_adjacent(p_qubits[0], p_qubits[1]));

  qubit_mapping[l_qubits[0]] = p_qubits[1];
  qubit_mapping[l_qubits[1]] = p_qubits[0];

  int temp = qubit_mask[p_qubits[0]];
  qubit_mask[p_qubits[0]] = qubit_mask[p_qubits[1]];
  qubit_mask[p_qubits[1]] = temp;

  mcts::MCTSNode* child_node = this->mcts_node_pool.create();

  *child_node = std::move(
      mcts::MCTSNode(&(this->coupling_graph), &(this->circuit), qubit_mapping,
                     qubit_mask, cur_node->front_layer, swap_gate, cur_node,
                     cur_node->probs_of_children[i]));
  if (this->extended)
    child_node->update_candidate_swap_list_extended();
  else
    child_node->update_candidate_swap_list();

  return child_node;
}

mcts::MCTSNode* mcts::MCTSTree::decide(mcts::MCTSNode* root_node) {
  float score = -1, child_value = 0.0;
  mcts::MCTSNode *res_node, *child = root_node->child;
  std::vector<int> visit_count_v(root_node->num_of_children, 0);
  std::vector<float> value_v(root_node->num_of_children, 0);
  for (int i = 0; i < root_node->num_of_children; i++) {
    child_value = this->gamma * child->value + child->reward;
    if (child_value > score) {
      res_node = child;
      score = child_value;
    }
    visit_count_v[i] = child->visit_count;
    value_v[i] = child->value;
    child = child->brother;
  }
  if (this->info) {
    for (int i = 0; i < root_node->num_of_children; i++) {
      std::cout << "{ " << visit_count_v[i] << "," << value_v[i] << "} ";
    }
    std::cout << std::endl;
  }
  return res_node;
}

void mcts::MCTSTree::fallback(mcts::MCTSNode* cur_node) {
  mcts::MCTSNode* child = cur_node->child;
  while (child != nullptr) {
    this->delete_tree(child);
    child = child->brother;
  }

  while (this->fallback_count > 0) {
    this->num_of_added_swap_gates -= 1;
    mcts::Gate swap_gate =
        this->coupling_graph.edges[this->swap_label_list.back()];
    this->swap_label_list.pop_back();
    cur_node->update_by_swap_gate(swap_gate);
    assert(cur_node->reward == 0);
    if (this->is_generate_data) {
      this->adj_list.pop_back();
      this->qubits_list.pop_back();
      this->num_list.pop_back();
      this->swap_label_list_n.pop_back();
      this->value_list.pop_back();
      this->action_prob_list.pop_back();
      this->num_of_samples -= 1;
    }
    this->fallback_count -= 1;
  }
  int s = -1, t = -1, m = this->coupling_graph.num_of_edges;
  for (auto& g : cur_node->front_layer) {
    std::vector<int> qubits = this->circuit.get_gate_qubits(g);
    int d = this->coupling_graph.distance(cur_node->qubit_mapping[qubits[0]],
                                          cur_node->qubit_mapping[qubits[1]]);
    if (d < m) {
      m = d;
      s = cur_node->qubit_mapping[qubits[0]];
      t = cur_node->qubit_mapping[qubits[1]];
    }
  }

  std::vector<int> sp = this->coupling_graph.shortest_path(s, t);
  for (int i = 0; i < sp.size() - 2; i++) {
    mcts::Gate sg(sp[i], sp[i + 1]);
    cur_node->update_by_swap_gate(sg);
    this->num_of_added_swap_gates += 1;
    this->swap_label_list.push_back(this->coupling_graph.swap_gate_label(sg));
    assert(cur_node->reward == 0 || (i == sp.size() - 3));
  }
  this->num_of_executed_gates += cur_node->reward;
  if (this->extended) {
    cur_node->update_candidate_swap_list_extended();
  } else {
    cur_node->update_candidate_swap_list();
  }

  if (!cur_node->is_terminal_node()) this->expand(cur_node);
}

void mcts::MCTSTree::generate_data(mcts::MCTSNode* cur_node,
                                   int swap_gate_label) {
  std::vector<int> subcircuit = cur_node->get_subcircuit(this->threshold_size);
  this->adj_list.emplace_back(std::move(
      this->circuit.get_adj_matrix(subcircuit, this->threshold_size)));
  this->qubits_list.emplace_back(std::move(this->circuit.get_qubits_matrix(
      subcircuit, cur_node->qubit_mapping, this->threshold_size)));
  this->num_list.emplace_back(subcircuit.size());
  this->swap_label_list_n.push_back(swap_gate_label);
  this->value_list.push_back(cur_node->value);
  this->action_prob_list.push_back(cur_node->probs_of_children);
  this->num_of_samples += 1;
}

int mcts::MCTSTree::get_num_of_samples() { return this->num_of_samples; }

std::vector<int> mcts::MCTSTree::get_added_swap_label_list() {
  return this->swap_label_list;
}

int* mcts::MCTSTree::get_adj_list() {
  int n = this->threshold_size * 5;
  this->adj_mem = new int[n * this->num_of_samples];
  for (int i = 0; i < this->num_of_samples; i++) {
    memcpy(static_cast<void*>(this->adj_mem + i * n),
           static_cast<void*>(this->adj_list[i].data()), sizeof(int) * n);
  }
  return this->adj_mem;
}
int* mcts::MCTSTree::get_qubits_list() {
  this->is_send_data = true;
  int n = this->threshold_size * 2;
  this->qubits_mem = new int[n * this->num_of_samples];
  for (int i = 0; i < this->num_of_samples; i++) {
    memcpy(static_cast<void*>(this->qubits_mem + i * n),
           static_cast<void*>(this->qubits_list[i].data()), sizeof(int) * n);
  }
  return this->qubits_mem;
}
int* mcts::MCTSTree::get_num_list() {
  this->num_mem = new int[this->num_of_samples];
  memcpy(static_cast<void*>(this->num_mem),
         static_cast<void*>(this->num_list.data()),
         sizeof(int) * this->num_of_samples);
  return this->num_mem;
}
int* mcts::MCTSTree::get_swap_label_list() {
  this->swap_label_mem = new int[this->num_of_samples];
  memcpy(static_cast<void*>(swap_label_mem),
         static_cast<void*>(this->swap_label_list_n.data()),
         sizeof(int) * this->num_of_samples);
  return this->swap_label_mem;
}
float* mcts::MCTSTree::get_value_list() {
  this->value_mem = new float[this->num_of_samples];
  memcpy(static_cast<void*>(value_mem),
         static_cast<void*>(this->value_list.data()),
         sizeof(float) * this->num_of_samples);
  return this->value_mem;
}
float* mcts::MCTSTree::get_action_prob_list() {
  int n = this->coupling_graph.num_of_edges;
  this->action_prob_mem = new float[n * this->num_of_samples];
  for (int i = 0; i < this->num_of_samples; i++) {
    memcpy(static_cast<void*>(this->action_prob_mem + i * n),
           static_cast<void*>(this->action_prob_list[i].data()),
           sizeof(float) * n);
  }
  return this->action_prob_mem;
}
