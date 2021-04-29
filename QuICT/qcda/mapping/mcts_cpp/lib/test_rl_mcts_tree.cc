#include<iostream>
#include<fstream>
#include<rl_mcts_tree.h>

int main(int argc, char* argv[]){
    std::string  input_file_path = argv[1]; 
    int num_of_gates, num_of_edges, num_of_qubits, feature_dim, front_layer_size;
    int *adj_matrix, *qubits_matrix, *distance_matrix, *label_matrix,
        *circuit, *dependency_graph;
    float *feature_matrix;
    std::ifstream adj_file(std::string(input_file_path).append("/distance_matrix.txt").c_str(), std::ios::binary),
                 distance_file(std::string(input_file_path).append("/distance_matrix.txt").c_str(), std::ios::binary),
                 feature_file(std::string(input_file_path).append("/feature_matrix.txt").c_str(), std::ios::binary),
                 label_file(std::string(input_file_path).append("/label_matrix.txt").c_str(), std::ios::binary),
                 circuit_file(std::string(input_file_path).append("/circuit.txt").c_str(), std::ios::binary),
                 dependency_file(std::string(input_file_path).append("/dependency_graph.txt").c_str(), std::ios::binary),
                 qubit_mapping_file(std::string(input_file_path).append("/qubit_mapping.txt").c_str(), std::ios::binary),
                 front_layer_file(std::string(input_file_path).append("/front_layer.txt").c_str(), std::ios::binary),
                 qubit_mask_file(std::string(input_file_path).append("/qubit_mask.txt").c_str(), std::ios::binary),
                 metadata(std::string(input_file_path).append("/metadata.txt").c_str());

    metadata>>num_of_qubits>>num_of_edges>>feature_dim>>num_of_gates>>front_layer_size;
    adj_matrix = new int[num_of_qubits * num_of_qubits];
    adj_file.read((char*)(adj_matrix), sizeof(int) * num_of_qubits * num_of_qubits);

    distance_matrix = new int[num_of_qubits * num_of_qubits];
    distance_file.read((char*)(distance_matrix), sizeof(int) * num_of_qubits * num_of_qubits);
    
    label_matrix = new int[num_of_qubits * num_of_qubits];
    label_file.read((char*)(label_matrix), sizeof(int) * num_of_qubits * num_of_qubits);

    
    feature_matrix = new float[num_of_qubits * feature_dim];
    feature_file.read((char*)(feature_matrix), sizeof(int) * num_of_qubits * feature_dim);

    
    circuit = new int[num_of_gates * 2];
    circuit_file.read((char*)(circuit), sizeof(int) * num_of_gates * 2);

    dependency_graph = new int[num_of_gates * 4];
    dependency_file.read((char*)(dependency_graph), sizeof(int) * num_of_gates * 4);

    std::vector<int> qubit_mapping(num_of_qubits, -1), qubit_mask(num_of_qubits, -1), front_layer(front_layer_size, -1);

    qubit_mapping_file.read((char*)(qubit_mapping.data()), sizeof(int)*num_of_qubits);
    qubit_mask_file.read((char*)(qubit_mask.data()), sizeof(int)*num_of_qubits);
    front_layer_file.read((char*)(front_layer.data()), sizeof(int)*front_layer_size);

    float gamma = 0.8, c = 20;
    bool with_predictor = false, extended = false, is_generate_data= false;
    int num_of_swap_gates = 15, num_of_process = 5, threshold_size = 150, virtual_loss = 0,
        num_of_iterations = 200, num_of_playout = 2, bp_mode = 0, size_of_subcircuits = 30;  
    char*  model_file_path;
    int device = 0, info = 0, major = 1 ; 
    gamma = std::stof(std::string(argv[2]));
    c = std::stof(std::string(argv[3]));

    with_predictor = std::stoi(std::string(argv[4]));
    extended = std::stoi(std::string(argv[5]));
    is_generate_data = std::stoi(std::string(argv[6]));

    num_of_swap_gates = std::stoi(std::string(argv[7]));
    num_of_process = std::stoi(std::string(argv[8]));
    threshold_size = std::stoi(std::string(argv[9]));
    virtual_loss = std::stoi(std::string(argv[10]));
    num_of_iterations = std::stoi(std::string(argv[11]));
    num_of_playout = std::stoi(std::string(argv[12]));
    bp_mode = std::stoi(std::string(argv[13]));  
    model_file_path = argv[14];
    device = std::stoi(std::string(argv[15]));
    info = std::stoi(std::string(argv[16])); 

    mcts::RLMCTSTree rl_mcts(major, info, extended, with_predictor, 
                is_generate_data, threshold_size,
                gamma, c, virtual_loss,
                bp_mode, num_of_process, 
                size_of_subcircuits, num_of_swap_gates, 
                num_of_iterations, num_of_playout, 
                num_of_qubits, feature_dim, num_of_edges,
                adj_matrix, distance_matrix, 
                label_matrix, feature_matrix,
                model_file_path, device);

    rl_mcts.load_data(num_of_gates, num_of_qubits, circuit, dependency_graph);
    rl_mcts.build_search_tree(qubit_mapping, qubit_mask, front_layer);
    std::vector<int> res = rl_mcts.run_rl();
    std::cout<<res[0]<<"   "<<res[1]<<std::endl;
    return 0;
}