#include<utility.h>
#include<queue>

mcts::CouplingGraph::CouplingGraph(){
    this->num_of_edges = 0;
    this->num_of_qubits = 0;
}



mcts::CouplingGraph::CouplingGraph(int num_of_qubits, int feature_dim, int num_of_edges, int * adj_matrix, int * distance_matrix, int * edge_label, float * feature_matrix){
    this->num_of_qubits = num_of_qubits;
    this->num_of_edges = num_of_edges;
    

    this->adj_matrix_accessor = adj_matrix;
    this->distance_matrix_accessor =  distance_matrix;
    this->edge_label_accessor = edge_label;
    this->feature_matrix_accessor = feature_matrix;



    this->edges.resize(this->num_of_edges);
    for(int i = 0; i < this->num_of_qubits; i++){
        for(int j = 0; j < this->num_of_qubits; j++){
            int idx = this->edge_label_accessor[i+j*this->num_of_qubits];
            if( idx != -1){
                this->edges[idx] = Gate(i,j);
            }
        }
    }
}


bool mcts::CouplingGraph::is_adjacent(int v, int u){
    assert(v < this->num_of_qubits && v >= 0);
    assert(u < this->num_of_qubits && u >= 0);
    return this->adj_matrix_accessor[this->num_of_qubits *v + u] == 1;
}

int mcts::CouplingGraph::distance(int v, int u){
    assert(v < this->num_of_qubits && v>=0);
    assert(u < this->num_of_qubits && u>=0);
    return this->distance_matrix_accessor[this->num_of_qubits *v +u];
}

int mcts::CouplingGraph::swap_gate_label(mcts::Gate& swap_gate){
    assert(this->is_adjacent(swap_gate.ctrl, swap_gate.tar));
    int c = swap_gate.ctrl, t = swap_gate.tar;
    return this->edge_label_accessor[this->num_of_qubits *c + t];

}

std::vector<int> mcts::CouplingGraph::shortest_path(int s, int t){
    assert(s < this->num_of_qubits && s>=0);
    assert(t < this->num_of_qubits && t>=0);
    std::vector<int> res;
    std::vector<int> parent(this->num_of_qubits, -1), 
                     mark(this->num_of_qubits, 1),
                     dis(this->num_of_qubits, 0);
    std::queue<int> q;
    q.push(s);
    mark[s] = 0;
    while(!q.empty()){
        int v = q.front();
        q.pop();
        for(int u = 0; u < this->num_of_qubits; u++){
            if(this->is_adjacent(v,u)){
                if(u == t){
                    int p = u; 
                    parent[p] = v;
                    while(p != -1){
                        res.push_back(p);
                        p = parent[p];
                    }
                    return res;
                }else if(mark[u]){
                    q.push(u);
                    mark[u] = 0;
                    parent[u] = v;
                    dis[u] = dis[v] + 1;
                    assert(dis[u] == this->distance(s,u));
                }
            }
        }
    }
    return res;
}




mcts::Circuit::Circuit(){
    this->num_of_gates = 0;
}


mcts::Circuit::Circuit(int num_of_gates, int * circuit, int * dependency_graph){
    this->num_of_gates = num_of_gates;

    this->circuit_accessor = circuit;
    this->dependency_graph_accessor = dependency_graph;
}

std::vector<int> mcts::Circuit::get_succeed_gates(int gate){
    assert(gate < this->num_of_gates);
    std::vector<int> res{this->dependency_graph_accessor[gate*4],
                        this->dependency_graph_accessor[gate*4+1]};
   
    return res;
}

std::vector<int> mcts::Circuit::get_gate_qubits(int gate){
    assert(gate < this->num_of_gates);
    std::vector<int> res{this->circuit_accessor[gate*2], 
                        this->circuit_accessor[gate*2+1]};
    return res;
}


std::vector<int> mcts::Circuit::get_adj_matrix(std::vector<int>& gates, int length){
    assert(gates.size() <= length);
    std::vector<int> idx(this->num_of_gates, -1); 
    std::vector<int> res(length * 5, -1);
    for(int i = 0; i < gates.size(); i++){
        idx[gates[i]] = i;
    }
    for(int i = 0; i < gates.size(); i++){
        int g = gates[i];
        res[5*i] = i;
        for(int j = 0; j < 4; j++){
            int t = this->dependency_graph_accessor[4*g + j];
            if(t == -1){
                res[5*i + j + 1] = -1;
            }else{
                res[5*i + j + 1] = idx[t];
            }
        }
    }
    return res;
}

std::vector<int> mcts::Circuit::get_qubits_matrix(std::vector<int>& gates, std::vector<int>& qubit_mapping, int length){
    assert(gates.size() <= length);
    std::vector<int> res(length * 2, -1);
    for(int i = 0; i < gates.size(); i++){
        int g = gates[i];
        res[2*i] = qubit_mapping[this->circuit_accessor[2*g]];
        res[2*i+1] = qubit_mapping[this->circuit_accessor[2*g+1]];
    }
    return res;
}




