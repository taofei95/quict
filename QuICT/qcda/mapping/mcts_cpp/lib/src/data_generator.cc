 #include<data_generator.h>
 
 mcts::DataGenerator::DataGenerator(int major, int info, bool with_predictor, 
                        bool is_generate_data, int threshold_of_circuit, 
                        float gamma, float c, 
                        float virtual_loss, int bp_mode, 
                        int num_of_process, int size_of_subcircuits, 
                        int num_of_swap_gates, int num_of_iterations, 
                        int num_of_playout, int num_of_qubits, 
                        int feature_dim, int num_of_edges, 
                        bool extended, int * coupling_graph, 
                        int * distance_matrix, int * edge_label, 
                        float * feature_matrix, const char* model_file_path, 
                        int device, int num_of_circuit_process, 
                        int inference_batch_size, int max_batch_size){
    
    this->major = major;
    this->info = info;
    this->model_file_path = model_file_path;
    this->device = device;
    this->inference_batch_size = inference_batch_size;
    this->max_batch_size = max_batch_size;
    this->num_of_circuit_process = num_of_circuit_process; 
    this->with_predictor = with_predictor;
    this->extended = extended;
    this->num_of_samples = 0;
    this->num_of_circuit = 0;
    this->with_predictor = with_predictor;
    this->is_generate_data = is_generate_data;
    this->threshold_size  = threshold_of_circuit;
    this->virtual_loss = virtual_loss;
    this->num_of_process = num_of_process;
    this->bp_mode = bp_mode;
    this->gamma = gamma;
    this->c = c;
    this->num_of_iterations = num_of_iterations;
    this->num_of_playout = num_of_playout;
    this->num_of_qubits = num_of_qubits;
    this->size_of_subcircuits = size_of_subcircuits;
    this->num_of_swap_gates =num_of_swap_gates;
    this->pool_lock.clear(std::memory_order_relaxed);
    this->coupling_graph = std::move(mcts::CouplingGraph(num_of_qubits, feature_dim, num_of_edges,
                             coupling_graph, distance_matrix, edge_label, feature_matrix));  
    
    this->res_flags = new std::atomic_flag[this->num_of_process*this->num_of_circuit_process];
    this->res_values = new float[this->num_of_process*this->num_of_circuit_process];
    this->res_probs = new torch::Tensor[this->num_of_process*this->num_of_circuit_process];
    this->thread_flag = new std::atomic_flag;
    this->thread_flag->test_and_set(std::memory_order_release);

    this->adj_mem = new int[max_batch_size * threshold_size * 5];
    this->qubits_mem = new int[max_batch_size * threshold_size * 2];
    this->num_mem = new int[max_batch_size ];
    this->swap_label_mem = new int[max_batch_size];
    this->value_mem = new float[max_batch_size ];
    this->action_prob_mem = new float[max_batch_size * num_of_edges];

    this->inferencer = std::move(std::thread(mcts::inference_thread, std::ref(this->sample_queue), this->thread_flag,
                            this->num_of_process, this->inference_batch_size, this->model_file_path, this->device,
                            this->res_flags, this->res_values, this->res_probs));

}


mcts::DataGenerator::~DataGenerator(){
    this->thread_flag->clear();
    this->inferencer.join();
    delete[] this->res_flags, 
           this->res_values, 
           this->res_probs,
           this->adj_mem,
           this->qubits_mem,
           this->num_mem,
           this->swap_label_mem,
           this->value_mem,
           this->qubits_mem;
    delete this->thread_flag;

}

void mcts::DataGenerator::load_data(int num_of_gates, int num_of_logical_qubits, 
                        int * circuit, int * dependency_graph, 
                        std::vector<int> qubit_mapping, std::vector<int> qubit_mask, 
                        std::vector<int> front_layer){
    this->circuit_list.emplace_back(std::move(mcts::Circuit(num_of_gates, circuit, dependency_graph)));
    this->qubit_mapping_list.emplace_back(std::move(qubit_mapping));
    this->qubit_mask_list.emplace_back(std::move(qubit_mask));
    this->front_layer_list.emplace_back(std::move(front_layer));
    this->num_of_circuit += 1;
}



void mcts::DataGenerator::run(){
    std::atomic_int id(0);
    int cid = 0;
    //omp_set_num_threads(this->num_of_process * this->num_of_circuit_process);
    int num_thread = std::min(this->num_of_circuit_process, this->num_of_circuit);
    #pragma omp parallel shared(id) private(cid) num_threads(num_thread)
    {
        cid = id.fetch_add(1);
        #pragma omp for
        for(int i = 0; i < this->num_of_circuit; i++){
            mcts::RLMCTSTree rl_mcts(this->major, this->info, this->extended, this->with_predictor, 
                                    this->is_generate_data, this->threshold_size,
                                    this->gamma, this->c, this->virtual_loss, 
                                    this->bp_mode, this->num_of_process, 
                                    this->size_of_subcircuits, this->num_of_swap_gates, 
                                    this->num_of_iterations, this->num_of_playout, 
                                    this->coupling_graph, cid,
                                    &(this->sample_queue), this->res_flags,
                                    this->res_values, this->res_probs);

            rl_mcts.load_data(this->num_of_qubits, this->circuit_list[i]);    
            rl_mcts.build_search_tree(this->qubit_mapping_list[i], 
                                     this->qubit_mask_list[i],
                                     this->front_layer_list[i]);  
            rl_mcts.run_rl();
            while(this->pool_lock.test_and_set(std::memory_order_acquire));
            assert(rl_mcts.num_of_samples == rl_mcts.adj_list.size());
            this->num_of_samples.fetch_add(rl_mcts.num_of_samples);
            
            this->adj_list.insert(this->adj_list.end(), rl_mcts.adj_list.begin(), rl_mcts.adj_list.end());
            this->qubits_list.insert(this->qubits_list.end(), rl_mcts.qubits_list.begin(), rl_mcts.qubits_list.end());
            this->num_list.insert(this->num_list.end(), rl_mcts.num_list.begin(), rl_mcts.num_list.end());
            this->swap_label_list.insert(this->swap_label_list.end(), rl_mcts.swap_label_list_n.begin(), rl_mcts.swap_label_list_n.end());
            this->value_list.insert(this->value_list.end(), rl_mcts.value_list.begin(), rl_mcts.value_list.end());
            this->action_prob_list.insert(this->action_prob_list.end(), rl_mcts.action_prob_list.begin(), rl_mcts.action_prob_list.end());
            this->pool_lock.clear(std::memory_order_release);            
        }
    }
}

void mcts::DataGenerator::clear(){
    this->circuit_list.clear();
    this->qubit_mapping_list.clear();
    this->qubit_mask_list.clear();
    this->front_layer_list.clear();
    this->num_of_circuit = 0;
}

void mcts::DataGenerator::update_model(){
    this->thread_flag->clear();
    this->inferencer.join();
    this->thread_flag->test_and_set(std::memory_order_release);
    this->inferencer = std::move(std::thread(inference_thread, std::ref(this->sample_queue), this->thread_flag,
                            this->num_of_process, this->inference_batch_size, this->model_file_path, this->device,
                            this->res_flags, this->res_values, this->res_probs));
}

int* mcts::DataGenerator::get_adj_list(std::vector<int>& batch_samples){
    int n = this->threshold_size * 5;
    for(int i = 0; i < batch_samples.size(); i++){
        memcpy(static_cast<void*>(this->adj_mem + i*n), 
            static_cast<void*>(this->adj_list[batch_samples[i]].data()),
            sizeof(int) * n);
    }
    return this->adj_mem;
}

int* mcts::DataGenerator::get_qubits_list(std::vector<int>& batch_samples){
    int n = this->threshold_size * 2;
    for(int i = 0; i < batch_samples.size(); i++){
        memcpy(static_cast<void*>(this->qubits_mem + i*n), 
            static_cast<void*>(this->qubits_list[batch_samples[i]].data()),
            sizeof(int) * n);
    }
    return this->qubits_mem;
}

int* mcts::DataGenerator::get_num_list(std::vector<int>& batch_samples){
    for(int i = 0; i < batch_samples.size(); i++){
        this->num_mem[i] = this->num_list[batch_samples[i]];
    }
    return this->num_mem;
}

int* mcts::DataGenerator::get_swap_label_list(std::vector<int>& batch_samples){
    for(int i = 0; i < batch_samples.size(); i++){
        this->swap_label_mem[i] = this->swap_label_list[batch_samples[i]];
    }
    return this->swap_label_mem;
}

float* mcts::DataGenerator::get_value_list(std::vector<int>& batch_samples){
    for(int i = 0; i < batch_samples.size(); i++){
        this->value_mem[i] = this->value_list[batch_samples[i]];
    }
    return this->value_mem;
}

float* mcts::DataGenerator::get_action_prob_list(std::vector<int>& batch_samples){
    int n = this->coupling_graph.num_of_edges;
    for(int i = 0; i < batch_samples.size(); i++){
        memcpy(static_cast<void*>(this->action_prob_mem + i*n), 
            static_cast<void*>(this->action_prob_list[batch_samples[i]].data()),
            sizeof(float) * n);
    }
    return this->action_prob_mem;
}

int mcts::DataGenerator::get_num_of_samples(){
    return this->num_of_samples.load();
}