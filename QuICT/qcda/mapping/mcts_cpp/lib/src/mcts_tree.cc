#include <mcts_tree.h>
#include <iostream>

float random_generator(int seed){
    std::mt19937_64 gen = std::mt19937_64((unsigned int)(time(NULL))*seed);
    std::uniform_real_distribution<float> dist = std::uniform_real_distribution<float>(0.0,1.0);
    return dist(gen);
}

int  choice(std::vector<float>& p, int seed){
    float r = random_generator(seed), t = 0;
    for(unsigned int i = 0; i < p.size(); i++){
        t += p[i];
        if(t>r){
            return i;
        }
    }
    return p.size() -1;
}


mcts::MCTSTree::MCTSTree(){
    this->gamma = 0.7;
    this->c = 20;
    this->num_of_iterations = 40;
    this->num_of_playout = 500;
    this->size_of_subcircuits = 30;
    
}

mcts::MCTSTree::MCTSTree(float gamma, float c, int size_of_subcircuits, int num_of_iterations, int num_of_playout, int num_of_qubits, int feature_dim, 
                        int num_of_edges, int * coupling_graph, int * distance_matrix, int *edge_label, float * feature_matrix){
    this->gamma = gamma;
    this->c = c;
    this->num_of_iterations = num_of_iterations;
    this->num_of_playout = num_of_playout;
    this->num_of_qubits = num_of_qubits;
    this->feature_dim = feature_dim;
    this->size_of_subcircuits = size_of_subcircuits;

    this->coupling_graph = std::make_shared<mcts::CouplingGraph>(num_of_qubits, feature_dim, num_of_edges, coupling_graph, distance_matrix, edge_label, feature_matrix);

    this->thread_pool = std::make_shared<ctpl::thread_pool>(16);
}
void mcts::MCTSTree::load_data(int num_of_gates, int num_of_logical_qubits, int * circuit, int * dependency_graph){
    this->num_of_gates = num_of_gates;
    this->num_of_logical_qubits = num_of_logical_qubits;
    this->circuit = std::make_shared<Circuit>(num_of_gates, circuit, dependency_graph);
}

void mcts::MCTSTree::build_search_tree(std::vector<int>& qubit_mapping, std::vector<int>& qubit_mask, std::vector<int>& front_layer){
    
    // std::cout<<"front_layer"<<std::endl;
    // for(auto & g : front_layer){
    //     std::cout<<g<<"  ";
    // }
    // std::cout<<std::endl;
    // std::cout<<"qubit mapping"<<std::endl;
    // for(auto & g : qubit_mapping){
    //     std::cout<<g<<"  ";
    // }
    // std::cout<<std::endl;

    // std::cout<<"qubit mask"<<std::endl;
    // for(auto & g : qubit_mask){
    //     std::cout<<g<<"  ";
    // }
    // std::cout<<std::endl;
    this->root_node = std::make_shared<mcts::MCTSNode>(this->coupling_graph,
                                                       this->circuit,
                                                       qubit_mapping,
                                                       qubit_mask,
                                                       front_layer,
                                                       mcts::Gate(),
                                                       std::move(std::shared_ptr<mcts::MCTSNode>(nullptr)),
                                                       0.0
                                                    );
    // std::cout<<"front_layer"<<std::endl;
    // for(auto & g : this->root_node->front_layer){
    //     std::cout<<g<<"  ";
    // }
    // std::cout<<std::endl;
    // std::cout<<"qubit mapping"<<std::endl;
    // for(auto & g : this->root_node->qubit_mapping){
    //     std::cout<<g<<"  ";
    // }
    // std::cout<<std::endl;

    // std::cout<<"qubit mask"<<std::endl;
    // for(auto & g : this->root_node->qubit_mask){
    //     std::cout<<g<<"  ";
    // }
    // std::cout<<std::endl;
   //float f = simulate_thread(0,*this->root_node, 1, this->size_of_subcircuits, this->gamma);
}

void mcts::MCTSTree::print_(){
    this->coupling_graph->print();
    this->circuit->print();
}

std::vector<int> mcts::MCTSTree::search_by_step(){
    //std::cout<<1<<std::endl;
    auto start = std::chrono::system_clock::now();
    this->search(this->root_node);
    std::shared_ptr<MCTSNode> best_child = this->decide(this->root_node);
    auto end   = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout<<"The time cost by the search: "<<(double)(duration.count())<<std::endl;
    
    std::vector<int> res {best_child->swap_gate.ctrl,best_child->swap_gate.tar};
    return res;
}

void mcts::MCTSTree::search(std::shared_ptr<mcts::MCTSNode> root_node){
    for(int i = 0; i < this->num_of_iterations; i++){
        this->search_thread(root_node);
    }
    // for(auto& child : root_node->children){
    //     std::cout<<child->value<<"  "<<std::endl;
    // }
    // std::cout<<std::endl;
}


void mcts::MCTSTree::search_thread(std::shared_ptr<mcts::MCTSNode> root_node){
    std::shared_ptr<mcts::MCTSNode> cur_node = this->select(root_node);
    this->expand(cur_node);
    float value = this->rollout(cur_node);
    //float value = 20;
    //std::cout<<"rollout"<<std::endl;
    //simulate_thread(0, *root_node, 1, this->size_of_subcircuits, this->gamma);
    this->backpropagate(cur_node->parent, value);
}

std::shared_ptr<mcts::MCTSNode> mcts::MCTSTree::select(std::shared_ptr<mcts::MCTSNode> root_node){
    std::shared_ptr<mcts::MCTSNode> cur_node = root_node;
    // std::cout<<"into selection"<<std::endl;
    while(!cur_node->is_leaf_node()){
        cur_node = this->select_next_child(cur_node);
    }
    // std::cout<<"out selection"<<std::endl;
    return cur_node;
}

std::shared_ptr<mcts::MCTSNode> mcts::MCTSTree::select_next_child(std::shared_ptr<mcts::MCTSNode> cur_node){
    float score = -1;
    std::shared_ptr<mcts::MCTSNode> res_node;
    for(auto& n : cur_node->children){
        float ucb = this->upper_condfidence_bound_with_predictor(n);
        if(ucb > score){
            res_node = n;
            score = ucb;
        }
    }
    return res_node;
}


float mcts::MCTSTree::upper_condfidence_bound(std::shared_ptr<mcts::MCTSNode> cur_node){
    return cur_node->value + this->c * sqrt(log((float)(cur_node->parent->visit_count)) / (cur_node->visit_count+ 0.001)); 
}

float mcts::MCTSTree::upper_condfidence_bound_with_predictor(std::shared_ptr<mcts::MCTSNode> cur_node){
    return cur_node->value + this->c * cur_node->prob * sqrt((float)(cur_node->parent->visit_count)/ (1 + cur_node->visit_count));
}



void mcts::MCTSTree::expand(std::shared_ptr<mcts::MCTSNode> cur_node){
    assert(cur_node->candidate_swap_list.size() ==  cur_node->probs_of_children.size());
    int num_of_swap_gates = cur_node->candidate_swap_list.size();
    // std::cout<<"into expansion"<<std::endl;
    for(int i = 0; i < num_of_swap_gates; i++){
        cur_node->add_child_node_by_swap_gate(cur_node->candidate_swap_list[i], cur_node->probs_of_children[i]);
    }
    // std::cout<<"out expansion"<<std::endl;
}

void mcts::MCTSTree::backpropagate(std::shared_ptr<mcts::MCTSNode> cur_node, float value){
    float bp_value = value;
    bool flag = true;
    //std::cout<<"backpropagate"<<std::endl;
    while(flag && cur_node.get() != nullptr){
        //std::cout<<"backpropagate"<<std::endl;
        bp_value = cur_node->reward + this->gamma * bp_value;
        if(bp_value > cur_node->value){
            cur_node->value = bp_value;
            cur_node = cur_node->parent;
            // std::cout<< new_value<<std::endl;
        }else{
            flag = false;
        }
    }
}

float mcts::MCTSTree::rollout(std::shared_ptr<mcts::MCTSNode> cur_node){
    return this->simulate_multithread(cur_node);
}

float mcts::MCTSTree::simulate(std::shared_ptr<MCTSNode> cur_node){
    std::vector<float> res(this->num_of_playout);
    
    #pragma omp parallel for
    for(int i = 0; i < this->num_of_playout; i++){
        res[i] = simulate_thread(0, *cur_node, i, this->size_of_subcircuits, this->gamma);
    }

    float maximum = -1;
    for(int i = 0; i < this->num_of_playout; i++){
        if(res[i] >maximum){
            //std::cout<<(maximum)<<std::endl;
            maximum = res[i];
        }
    }
    return maximum;
}

float mcts::MCTSTree::simulate_multithread(std::shared_ptr<mcts::MCTSNode> cur_node){
    //ctpl::thread_pool simulation_pool(16);
    std::vector<std::future<float>> thread_handler(this->num_of_playout);
    std::vector<float> res(this->num_of_playout);
    for(int i = 0; i < this->num_of_playout; i++){
        thread_handler[i] = this->thread_pool->push(mcts::simulate_thread, *cur_node, i, this->size_of_subcircuits, this->gamma);
    }
    // for(int i = 0; i < this->num_of_playout; i++){
    //     thread_handler[i].wait();
    // }
    for(int i = 0; i < this->num_of_playout; i++){
        res[i] = thread_handler[i].get();
        //std::cout<<res[i]<<std::endl;
    }
    float maximum = -1;
    for(int i = 0; i < this->num_of_playout; i++){
        if(res[i] >maximum){
            //std::cout<<(maximum)<<std::endl;
            maximum = res[i];
        }
    }
    return maximum;
}

float mcts::simulate_thread(int id, mcts::MCTSNode cur_node, int seed, int size_of_subcircuits, float gamma){
    float res = 0;
    int num_of_executed_gates = 0, num_of_swap_gates = 0;
    //std::cout<<1<<std::endl;
    while(num_of_executed_gates < size_of_subcircuits && !cur_node.is_terminal_node()){
        int idx = choice(cur_node.probs_of_children, seed);
        // std::cout<<"probability"<<std::endl;
        // for(auto& p : cur_node.probs_of_children){
        //     std::cout<<p<<"  ";
        // }
        // std::cout<<std::endl;
        // std::cout<<"front_layer"<<std::endl;
        // for(auto & g : cur_node.front_layer){
        //     std::cout<<g<<"  ";
        // }
        // std::cout<<std::endl;
        // std::cout<<"qubit mapping"<<std::endl;
        // for(auto & g : cur_node.qubit_mapping){
        //     std::cout<<g<<"  ";
        // }
        // std::cout<<std::endl;

        // std::cout<<"qubit mask"<<std::endl;
        // for(auto & g : cur_node.qubit_mask){
        //     std::cout<<g<<"  ";
        // }

        // std::cout<<std::endl;

        num_of_executed_gates += cur_node.update_by_swap_gate(cur_node.candidate_swap_list[idx]);
        num_of_swap_gates += 1;
 
        
        //std::cout<<num_of_executed_gates<<std::endl;
    }
    int num = 0;
    if(num_of_executed_gates < size_of_subcircuits){
        num = num_of_executed_gates;
    }else{
        num = size_of_subcircuits;
    }
    res = (float)(num) * pow(gamma, float(num_of_swap_gates)/2);
    return res;
}


std::shared_ptr<mcts::MCTSNode> mcts::MCTSTree::decide(std::shared_ptr<mcts::MCTSNode> root_node){
    float score = -1;
    std::shared_ptr<mcts::MCTSNode> res_node; 
    for(auto child : root_node->children){
        if(child->value > score){
            score = child->value;
            res_node = child;
        }
    }
    return res_node;
}

void mcts::MCTSTree::run(){
    

}








