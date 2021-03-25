#include<utility.h>

void mcts::print_tensor(torch::Tensor & data){
    std::cout<<data<<std::endl;
}

mcts::CouplingGraph::CouplingGraph(){
    this->num_of_edges = 0;
    this->num_of_qubits = 0;
}

mcts::CouplingGraph&  mcts::CouplingGraph::operator=(mcts::CouplingGraph && a){
    if(this != &a){
        this->num_of_edges = a.num_of_edges;
        this->num_of_qubits = a.num_of_qubits;

        
        


        this->adj_matrix = std::move(a.adj_matrix);
        this->distance_matrix = std::move(a.distance_matrix);
        this->edge_label = std::move(a.edge_label);
        this->feature_matrix = std::move(a.feature_matrix);

        this->adj_matrix_accessor = std::make_unique<torch::TensorAccessor<int, 2>>(this->adj_matrix.accessor<int,2>());
        this->distance_matrix_accessor =  std::make_unique<torch::TensorAccessor<int, 2>>(this->distance_matrix.accessor<int,2>());
        this->edge_label_accessor = std::make_unique<torch::TensorAccessor<int, 2>>(this->edge_label.accessor<int,2>());
        this->feature_matrix_accessor = std::make_unique<torch::TensorAccessor<float, 2>>(this->feature_matrix.accessor<float,2>());
    }
    return *this;
}

mcts::CouplingGraph::CouplingGraph(int num_of_qubits, int feature_dim, int num_of_edges, int * adj_matrix, int * distance_matrix, int * edge_label, float * feature_matrix){
    this->num_of_qubits = num_of_qubits;
    this->num_of_edges = num_of_edges;
    
    this->adj_matrix = torch::from_blob(adj_matrix, {num_of_qubits, num_of_qubits}, torch::dtype(torch::kInt32));
    this->distance_matrix = torch::from_blob(distance_matrix, {num_of_qubits, num_of_qubits}, torch::dtype(torch::kInt32));
    this->feature_matrix = torch::from_blob(feature_matrix, {num_of_qubits, feature_dim}, torch::dtype(torch::kFloat32));
    this->edge_label = torch::from_blob(edge_label, {num_of_qubits, num_of_qubits}, torch::dtype(torch::kInt32));  

    this->adj_matrix_accessor = std::make_unique<torch::TensorAccessor<int, 2>>(this->adj_matrix.accessor<int,2>());
    this->distance_matrix_accessor =  std::make_unique<torch::TensorAccessor<int, 2>>(this->distance_matrix.accessor<int,2>());
    this->edge_label_accessor = std::make_unique<torch::TensorAccessor<int, 2>>(this->edge_label.accessor<int,2>());
    this->feature_matrix_accessor = std::make_unique<torch::TensorAccessor<float, 2>>(this->feature_matrix.accessor<float,2>());
}

void mcts::CouplingGraph::print(){
    mcts::print_tensor(this->adj_matrix);
    mcts::print_tensor(this->distance_matrix);
    mcts::print_tensor(this->edge_label);
    mcts::print_tensor(this->feature_matrix);
}

bool mcts::CouplingGraph::is_adjacent(int v, int u){
    assert(v < this->num_of_qubits);
    assert(u < this->num_of_qubits);
    return (*this->adj_matrix_accessor)[v][u] == 1;
}

int mcts::CouplingGraph::distance(int v, int u){
    assert(v < this->num_of_qubits);
    assert(u < this->num_of_qubits);
    return (*this->distance_matrix_accessor)[v][u];
}




mcts::Circuit::Circuit(){
    this->num_of_gates = 0;
}

mcts::Circuit& mcts::Circuit::operator=(mcts::Circuit && a){
    if(this != &a){
        this->num_of_gates = a.num_of_gates;
        this->circuit = std::move(a.circuit);
        this->dependency_graph = std::move(a.dependency_graph);

        this->circuit_accessor = std::make_unique<torch::TensorAccessor<int,2>>(this->circuit.accessor<int,2>());
        this->dependency_graph_accessor = std::make_unique<torch::TensorAccessor<int,2>>(this->dependency_graph.accessor<int,2>());
 
    }
    return *this;
}

mcts::Circuit::Circuit(int num_of_gates, int * circuit, int * dependency_graph){
    this->num_of_gates = num_of_gates;
    this->circuit = torch::from_blob(circuit, {num_of_gates,2}, torch::dtype(torch::kInt32));
    this->dependency_graph = torch::from_blob(dependency_graph, {num_of_gates, 4}, torch::dtype(torch::kInt32));

    this->circuit_accessor = std::make_unique<torch::TensorAccessor<int,2>>(this->circuit.accessor<int,2>());
    this->dependency_graph_accessor = std::make_unique<torch::TensorAccessor<int,2>>(this->dependency_graph.accessor<int,2>());
}

std::vector<int> mcts::Circuit::get_succeed_gates(int gate){
    assert(gate < this->num_of_gates);
    std::vector<int> res(2, 0);
    res[0] = (*this->dependency_graph_accessor)[gate][0];
    res[1] = (*this->dependency_graph_accessor)[gate][1];
    return res;
}

std::vector<int> mcts::Circuit::get_gate_qubits(int gate){
    assert(gate < this->num_of_gates);
    std::vector<int> res(2, 0);
    res[0] = (*this->circuit_accessor)[gate][0];
    res[1] = (*this->circuit_accessor)[gate][1];
    return res;
}

void mcts::Circuit::print(){
    for(int i = 0; i <num_of_gates; i++){
        for(int j = 0; j < 2; j++){
            std::cout<<(*this->circuit_accessor)[i][j]<<"  ";
        }
        std::cout<<std::endl;
    }
    // mcts::print_tensor(this->circuit);
    // mcts::print_tensor(this->dependency_graph);
}


