#include<mcts_node.h>

mcts::MCTSNode::MCTSNode(){
    this->value = 0.0;
    this->w = 0.0;
    this->reward = 0.0;
    this->prob = 0.0;
    this->visit_count = 0; 
    this->parent = nullptr;
    this->child = nullptr;
    this->brother =nullptr;
}

mcts::MCTSNode::MCTSNode(CouplingGraph* coupling_graph, 
                 Circuit* circuit,
                 std::vector<int> &qubit_mapping,
                 std::vector<int> &qubit_mask,
                 std::vector<int> &front_layer,
                 mcts::Gate swap_gate,
                 MCTSNode* parent,
                 float prob){
    this->value = 0.0;
    this->reward = 0.0;
    this->w = 0.0;
    this->prob = prob;
    this->visit_count = 0; 
    this->parent = parent;
    this->child = nullptr;
    this->brother =nullptr;
    this->swap_gate = swap_gate;
    this->num_of_gates = 0;

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
}

mcts::MCTSNode::MCTSNode(mcts::MCTSNode& node){
    this->value = node.value;
    this->w = node.w;
    this->reward = node.reward;
    this->prob = node.prob;
    this->visit_count = node.visit_count;

    this->num_of_children = node.num_of_children; 
    this->num_of_gates = node.num_of_gates;
    this->swap_gate = node.swap_gate; 
    this->parent = node.parent;
    this->brother = node.brother;
    this->child = node.child;

    this->front_layer = node.front_layer;
    this->qubit_mapping = node.qubit_mapping;
    this->inverse_qubit_mapping = node.inverse_qubit_mapping;
    this->qubit_mask = node.qubit_mask;
    
    this->candidate_swap_list = node.candidate_swap_list;
    this->probs_of_children = node.probs_of_children;
    this->coupling_graph = node.coupling_graph;
    this->circuit = node.circuit;
}

mcts::MCTSNode& mcts::MCTSNode::operator=(mcts::MCTSNode&& node){
        if(this != &node){
            this->value = node.value;
            this->w = node.w;
            this->reward = node.reward;
            this->prob = node.prob;
            this->visit_count = node.visit_count;

            this->num_of_children = node.num_of_children; 
            this->num_of_gates = node.num_of_gates;
            this->swap_gate = node.swap_gate; 
            this->parent = node.parent;
            this->brother = node.brother;
            this->child = node.child;

            this->front_layer = std::move(node.front_layer);
            this->qubit_mapping = std::move(node.qubit_mapping);
            this->inverse_qubit_mapping = std::move(node.inverse_qubit_mapping);
            this->qubit_mask = std::move(node.qubit_mask);
            this->candidate_swap_list = std::move(node.candidate_swap_list);
            this->probs_of_children = std::move(node.probs_of_children);
            this->candidate_swap_list = node.candidate_swap_list;
            this->probs_of_children = node.probs_of_children;
            this->coupling_graph = node.coupling_graph;
            this->circuit = node.circuit;
        }
        return *this;
}
void mcts::MCTSNode::clear(){
    this->candidate_swap_list.clear();
    this->candidate_swap_list.shrink_to_fit();
    this->probs_of_children.clear();
    this->probs_of_children.shrink_to_fit();
    this->front_layer.clear();
    this->front_layer.shrink_to_fit();
    this->qubit_mapping.clear();
    this->qubit_mapping.shrink_to_fit();
    this->qubit_mask.clear();
    this->qubit_mask.shrink_to_fit();
    this->inverse_qubit_mapping.clear();
    this->inverse_qubit_mapping.shrink_to_fit();
}
int  mcts::MCTSNode::update_front_layer(){
    int num_of_executed_gates = 0;
    std::vector<int> temp;
    temp.swap(this->front_layer);
    while(!temp.empty()){
        int gate = temp.back();
        temp.pop_back();
        if( this->is_gate_executable(gate) ){
            num_of_executed_gates += 1;
            std::vector<int> qubits = this->circuit->get_gate_qubits(gate);
            std::vector<int> successor = this->circuit->get_succeed_gates(gate); 
            this->qubit_mask[this->qubit_mapping[qubits[0]]] = successor[0];
            this->qubit_mask[this->qubit_mapping[qubits[1]]] = successor[1];
            for(auto &s:successor){
                if(s != -1){
                    if(this->is_gate_free(s)){
                        temp.push_back(s);
                    }
                }
            } 
        }else{
            this->front_layer.push_back(gate);
        }
    }
    this->reward = num_of_executed_gates;
    if(this->num_of_gates == 0){
        if(!this->parent){
            this->num_of_gates = this->circuit->num_of_gates - this->reward;
        }else{
            this->num_of_gates = this->parent->num_of_gates - this->reward;
        }
    }else{
        this->num_of_gates = this->num_of_gates - this->reward;
    }

    return num_of_executed_gates;
}   

std::vector<int> mcts::MCTSNode::get_subcircuit(int num_of_circuits){
    int n = std::min(num_of_circuits, this->num_of_gates);
    std::vector<int> res(n, 0), logical_qubit_mask(this->qubit_mapping.size(), -1);
    std::queue<int> q;
    for(int i = 0; i < this->qubit_mask.size(); i++){
        logical_qubit_mask[this->inverse_qubit_mapping[i]] = this->qubit_mask[i];
    }

    for(auto& p: this->front_layer){
        q.push(p);
    } 

    for(int i = 0; i < n; i++){
        assert(q.size()>0);
        int gate = q.front();
        //std::cout<<i<<std::endl;
        q.pop();
        res[i] = gate;
        std::vector<int> qubits = this->circuit->get_gate_qubits(gate);
        std::vector<int> successor = this->circuit->get_succeed_gates(gate); 
        logical_qubit_mask[qubits[0]] = successor[0];
        logical_qubit_mask[qubits[1]] = successor[1];
        for(auto &s:successor){
            if(s != -1){
                if(this->is_gate_free(s, logical_qubit_mask)){
                    q.push(s);
                }
            }
        } 
    }

    return res;
}

bool mcts::MCTSNode::is_gate_executable(int gate){
    std::vector<int> qubits = this->circuit->get_gate_qubits(gate);
    return this->coupling_graph->is_adjacent(this->qubit_mapping[qubits[0]],
                                             this->qubit_mapping[qubits[1]]);
}

bool mcts::MCTSNode::is_gate_free(int gate){
    std::vector<int> qubits = this->circuit->get_gate_qubits(gate);
    int ctrl = this->qubit_mask[this->qubit_mapping[qubits[0]]], targ = this->qubit_mask[this->qubit_mapping[qubits[1]]];
    return ( ctrl == -1 || ctrl == gate) && (targ == -1 ||  targ == gate);
}

bool mcts::MCTSNode::is_gate_free(int gate, std::vector<int>& logical_qubit_mask){
    std::vector<int> qubits = this->circuit->get_gate_qubits(gate);
    int ctrl = logical_qubit_mask[qubits[0]], targ = logical_qubit_mask[qubits[1]];
    return ( ctrl == -1 || ctrl == gate) && (targ == -1 ||  targ == gate);
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
    this->probs_of_children = std::move(std::vector<float>(num_of_swap_gates, 1.0 /(num_of_swap_gates+0.00001)));
    this->num_of_children = num_of_swap_gates;
}

void mcts::MCTSNode::update_candidate_swap_list_extended(){
    this->candidate_swap_list = this->coupling_graph->edges;
    this->probs_of_children = std::move(std::vector<float>(this->coupling_graph->num_of_edges, 1/(float)(this->coupling_graph->num_of_edges)));
    this->num_of_children = this->coupling_graph->num_of_edges;
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



bool mcts::MCTSNode::is_leaf_node(){
    if(this->child == nullptr){
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
    return res;
}





