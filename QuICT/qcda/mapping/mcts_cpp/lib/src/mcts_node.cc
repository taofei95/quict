#include<mcts_node.h>

mcts::MCTSNode::MCTSNode(){
    this->value = 0.0;
    this->reward = 0.0;
    this->prob = 0.0;
    this->visit_count = 0; 
    this->parent = std::shared_ptr<mcts::MCTSNode>(nullptr);
}

mcts::MCTSNode::MCTSNode(std::shared_ptr<mcts::CouplingGraph>& coupling_graph, 
                 std::shared_ptr<mcts::Circuit>& circuit,
                 std::vector<int> &qubit_mapping,
                 std::vector<int> &qubit_mask,
                 std::vector<int> &front_layer,
                 mcts::Gate swap_gate,
                 std::shared_ptr<mcts::MCTSNode>&& parent,
                 float prob){
    this->value = 0.0;
    this->reward = 0.0;
    this->prob = prob;
    this->visit_count = 0; 
    this->parent = parent;
    this->swap_gate = swap_gate;


    this->front_layer = front_layer;
    this->qubit_mapping = qubit_mapping;
    this->qubit_mask = qubit_mask;
    
    this->coupling_graph = coupling_graph;
    this->circuit = circuit;

    int num_of_physical_qubits = this->coupling_graph->num_of_qubits;
    this->inverse_qubit_mapping = std::move(std::vector<int>(num_of_physical_qubits, -1));
    for(unsigned int i = 0; i < this->qubit_mapping.size(); i++){
       this->inverse_qubit_mapping[this->qubit_mapping[i]] = i; 
    }

    this->update_front_layer();
    this->update_candidate_swap_list(); 
}

int  mcts::MCTSNode::update_front_layer(){
    int num_of_executed_gates = 0;
    std::vector<int> temp;
    temp.swap(this->front_layer);
    while(!temp.empty()){
        int gate = temp.back();
        // std::cout<<gate<<std::endl;
        temp.pop_back();
        if( this->is_gate_executable(gate) ){
            num_of_executed_gates += 1;
            std::vector<int> qubits = this->circuit->get_gate_qubits(gate);
            std::vector<int> successor = this->circuit->get_succeed_gates(gate); 
            this->qubit_mask[this->qubit_mapping[qubits[0]]] = successor[0];
            this->qubit_mask[this->qubit_mapping[qubits[1]]] = successor[1];
            for(auto &s:successor){
                // std::cout<<s<<std::endl;
                if(s != -1){
                    if(this->is_gate_free(s)){
                        // std::cout<<"free"<<std::endl;
                        temp.push_back(s);
                    }
                }
            } 
        }else{
            this->front_layer.push_back(gate);
        }
    }
    //this->front_layer.swap(temp);
    this->reward = num_of_executed_gates;
    this->value = reward;
    return num_of_executed_gates;
}   

bool mcts::MCTSNode::is_gate_executable(int gate){
    std::vector<int> qubits = this->circuit->get_gate_qubits(gate);
    return this->coupling_graph->is_adjacent(this->qubit_mapping[qubits[0]],
                                             this->qubit_mapping[qubits[1]]);
}

bool mcts::MCTSNode::is_gate_free(int gate){
    std::vector<int> qubits = this->circuit->get_gate_qubits(gate);
    int ctrl = this->qubit_mask[this->qubit_mapping[qubits[0]]], tar = this->qubit_mask[this->qubit_mapping[qubits[1]]];
    return ( ctrl == -1 || ctrl == gate) && (tar == -1 ||  ctrl == gate);
}

float f_(int x){
    if(x < 0){
        return 0.0;
    }else if(x == 0){
        return 0.001;
    }else{
        return (float)x;
    }
}

std::vector<float> f(std::vector<int>& x){
    std::vector<float> res(x.size(),0);
    for(int i = 0; i < x.size(); i++){
        res[i] = f_(x[i]);
    }
    return res;
} 

void mcts::MCTSNode::update_candidate_swap_list(){
    int num_of_qubits = this->coupling_graph->num_of_qubits;

    std::vector<int> qubits_list = std::move( this->get_invloved_qubits());
    std::vector<int> qubits_mark(num_of_qubits, 0);
    this->candidate_swap_list.clear();
    this->candidate_swap_list.reserve(this->coupling_graph->num_of_edges);
    
    std::sort(qubits_list.begin(), qubits_list.end());
    for(auto& i : qubits_list){
        qubits_mark[i] = 1;
        for(int j = 0; j < num_of_qubits; j++){
            if(qubits_mark[j] == 0 && this->coupling_graph->is_adjacent(i,j)){
                this->candidate_swap_list.push_back(mcts::Gate(i,j));
            }
        }
    }
    float sum = 0;
    unsigned int num_of_swap_gates = this->candidate_swap_list.size();
    
    this->probs_of_children.resize(num_of_swap_gates, 0);
    int base_cost = this->nearest_neighbour_cost(this->qubit_mapping);
    std::vector<int> nnc(num_of_swap_gates, 0);
    for(unsigned int i = 0; i < num_of_swap_gates; i++){
        nnc[i] = (base_cost - this->swap_cost(this->candidate_swap_list[i]));
    }
    this->probs_of_children = f(nnc);
   
    for(auto& p : this->probs_of_children){
        sum += p;
    }
    for(auto& p : this->probs_of_children){
        p = (p)/sum;
    }
}

std::vector<int> mcts::MCTSNode::get_invloved_qubits(){
    std::vector<int> qubits_list;
    qubits_list.reserve(this->coupling_graph->num_of_qubits);

    for(auto& g : this->front_layer){
        std::vector<int> qubits = this->circuit->get_gate_qubits(g);
        qubits_list.push_back(this->qubit_mapping[qubits[0]]);
        qubits_list.push_back(this->qubit_mapping[qubits[1]]);
    }
    return qubits_list;
}

int mcts::MCTSNode::nearest_neighbour_cost(std::vector<int>& cur_mapping){
    int res = 0;
    for(auto& g : this->front_layer){
        std::vector<int> qubits = this->circuit->get_gate_qubits(g);
        res += this->coupling_graph->distance(cur_mapping[qubits[0]], cur_mapping[qubits[1]]);
    }
    return res; 
}

int mcts::MCTSNode::swap_cost(mcts::Gate& swap_gate){
    int res = 0;
    std::vector<int> cur_mapping = this->qubit_mapping;
    std::vector<int> p_qubits = {swap_gate.ctrl, swap_gate.tar};
    std::vector<int> l_qubits = {this->inverse_qubit_mapping[swap_gate.ctrl], this->inverse_qubit_mapping[swap_gate.tar]};
    assert(this->coupling_graph->is_adjacent(p_qubits[0], p_qubits[1]));
    cur_mapping[l_qubits[0]] = p_qubits[1];
    cur_mapping[l_qubits[1]] = p_qubits[0];
    res = this->nearest_neighbour_cost(cur_mapping);
    return res;
}

bool mcts::MCTSNode::is_leaf_node(){
    if(this->children.empty()){
        return true;
    }else{
        return false;
    }
}

bool mcts::MCTSNode::is_terminal_node(){
    if(this->front_layer.empty()){
        return true;
    }else{
        return false;
    }
}


void mcts::MCTSNode::add_child_node_by_swap_gate(mcts::Gate& swap_gate, float prob){
    std::vector<int> p_qubits = {swap_gate.ctrl, swap_gate.tar};
    std::vector<int> l_qubits = {this->inverse_qubit_mapping[swap_gate.ctrl], this->inverse_qubit_mapping[swap_gate.tar]};

    std::vector<int> qubit_mapping = this->qubit_mapping;
    std::vector<int> qubit_mask = this->qubit_mask;

    assert(this->coupling_graph->is_adjacent(p_qubits[0], p_qubits[1]));
    

    qubit_mapping[l_qubits[0]] = p_qubits[1];
    qubit_mapping[l_qubits[1]] = p_qubits[0];
    

    int temp = qubit_mask[p_qubits[0]];
    qubit_mask[p_qubits[0]] = qubit_mask[p_qubits[1]];
    qubit_mask[p_qubits[1]] = temp;

    std::shared_ptr<mcts::MCTSNode> child = std::make_shared<mcts::MCTSNode>(this->coupling_graph,
                                   this->circuit,
                                   qubit_mapping,
                                   qubit_mask,
                                   this->front_layer,
                                   swap_gate,
                                   std::move(std::shared_ptr<mcts::MCTSNode>(this)),
                                   prob);
    this->children.push_back(child);
}

int mcts::MCTSNode::update_by_swap_gate(mcts::Gate& swap_gate){
    std::vector<int> p_qubits = {swap_gate.ctrl, swap_gate.tar};
    std::vector<int> l_qubits = {this->inverse_qubit_mapping[swap_gate.ctrl], this->inverse_qubit_mapping[swap_gate.tar]};

    assert(this->coupling_graph->is_adjacent(p_qubits[0], p_qubits[1]));
    
    this->qubit_mapping[l_qubits[0]] = p_qubits[1];
    this->qubit_mapping[l_qubits[1]] = p_qubits[0];
    
    this->inverse_qubit_mapping[p_qubits[0]] = l_qubits[1];
    this->inverse_qubit_mapping[p_qubits[1]] = l_qubits[0];

    int temp = this->qubit_mask[p_qubits[0]];
    this->qubit_mask[p_qubits[0]] = this->qubit_mask[p_qubits[1]];
    this->qubit_mask[p_qubits[1]] = temp;

    int res = this->update_front_layer();
    this->update_candidate_swap_list();
    return res;
}





