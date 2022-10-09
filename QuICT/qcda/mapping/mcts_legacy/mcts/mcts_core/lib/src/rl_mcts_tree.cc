 #include<rl_mcts_tree.h>

 void mcts::inference_thread(mcts::SampleQueue& sample_queue, std::atomic_flag* thread_flag, 
                        int num_of_process, int batch_size, std::string model_file_path, int device,
                        std::atomic_flag* res_flags, float* res_values, torch::Tensor* res_probs){
    mcts::Model nn_model = torch::jit::load(model_file_path.c_str());
    at::Device device_type = at::Device(torch::kCPU);
    if(device == 0){
        device_type = at::Device(torch::kCPU); 
    }else if(device<4){
        device_type = at::Device(torch::kCUDA, device-1);
    }else{
        assert(device>=4);
    }
    nn_model.to(device_type, at::kFloat);
    nn_model.eval();
    std::vector<torch::jit::IValue> inputs(3);
    mcts::Sample samples[batch_size];
    torch::Tensor adj_batch[batch_size], qubits_batch[batch_size], padding_mask_batch[batch_size];
    int ids[batch_size];
    int batch_count;
    while(thread_flag->test_and_set(std::memory_order_acquire)){
        batch_count = sample_queue.wait_dequeue_bulk_timed(samples, batch_size, 3);
        if(batch_count>0){
            for(int i = 0; i < batch_count; i++){
                adj_batch[i] = std::move(samples[i].adj);
                qubits_batch[i] = std::move(samples[i].qubits);
                padding_mask_batch[i] = std::move(samples[i].padding_mask);
                ids[i] = samples[i].cid*num_of_process + samples[i].pid;
            }
            inputs[0] = std::move(torch::stack(at::ArrayRef<torch::Tensor>(qubits_batch,batch_count), 0)).to(device_type, torch::kInt32);
            inputs[1] = std::move(torch::stack(at::ArrayRef<torch::Tensor>(padding_mask_batch, batch_count), 0)).to(device_type);
            inputs[2] = std::move(torch::stack(at::ArrayRef<torch::Tensor>(adj_batch, batch_count), 0)).to(device_type, torch::kInt32);

            auto outputs =  nn_model.forward(inputs).toTuple();
            auto values = outputs->elements()[1].toTensor();
            auto policy = outputs->elements()[0].toTensor().to(at::Device(torch::kCPU));
            torch::Tensor probs = torch::softmax(policy, 1); 
            using  namespace at::indexing;
            for(int i = 0; i < batch_count; i++){
                res_values[ids[i]] = values[i].item<float>();
                res_probs[ids[i]] = std::move(probs.index({i,"..."}));
                res_flags[ids[i]].clear(std::memory_order_release);
            }
        }
    }
}

mcts::RLMCTSTree::RLMCTSTree(int major, int info, bool extened, bool with_predictor,
            bool is_generate_data, int threshold_size, 
            float gamma, float c, float virtual_loss, 
            int bp_mode, int num_of_process, 
            int size_of_subcircuits,  int num_of_swap_gates, 
            int num_of_iterations, int num_of_playout,
            int num_of_qubits, int feature_dim, int num_of_edges, 
            int * coupling_graph, int * distance_matrix, 
            int * edge_label, float * feature_matrix,
            const char* model_file_path, int device):
            mcts::MCTSTree(major, 0,info,
                        extened, with_predictor,
                        is_generate_data, threshold_size, 
                        gamma, c, virtual_loss, 
                        bp_mode, num_of_process, 
                        size_of_subcircuits, num_of_swap_gates, 
                        num_of_iterations, num_of_playout,
                        num_of_qubits, feature_dim, num_of_edges, 
                        coupling_graph, distance_matrix, 
                        edge_label, feature_matrix){
        this->cid = 0;
        this->mode = 0;
        this->sample_queue = new mcts::SampleQueue();
        this->res_flags = new std::atomic_flag[this->num_of_process];
        this->res_values = new float[this->num_of_process];
        this->res_probs = new torch::Tensor[this->num_of_process];
        this->thread_flag = new std::atomic_flag;
        this->thread_flag->test_and_set(std::memory_order_release);
        this->inferencer = std::move(std::thread(inference_thread, std::ref(*(this->sample_queue)), this->thread_flag,
                                this->num_of_process, this->num_of_process, model_file_path, device,
                                this->res_flags, this->res_values, this->res_probs));

    }

mcts::RLMCTSTree::RLMCTSTree(int major, int info, bool extened, bool with_predictor,
                bool is_generate_data, int threshold_size, 
                float gamma, float c, float virtual_loss, 
                int bp_mode, int num_of_process, 
                int size_of_subcircuits,  int num_of_swap_gates, 
                int num_of_iterations, int num_of_playout,
                mcts::CouplingGraph& coupling_graph, int cid,
                mcts::SampleQueue* sample_queue, std::atomic_flag* res_flags,
                float* res_values, torch::Tensor* res_probs){
    this->major = major;
    this->info;
    this->bp_mode = bp_mode;
    this->with_predictor = with_predictor;
    this->extended = extended;
    this->num_of_samples = 0;
    this->fallback_count = 0;
    this->num_of_added_swap_gates = 0;
    this->num_of_executed_gates = 0;
    this->with_predictor = with_predictor;
    this->is_generate_data = is_generate_data;
    this->threshold_size  = threshold_size;
    this->virtual_loss = virtual_loss;
    this->num_of_process = num_of_process;
    this->gamma = gamma;
    this->c = c;
    this->num_of_iterations = num_of_iterations;
    this->num_of_playout = num_of_playout;
    this->size_of_subcircuits = size_of_subcircuits;
    this->num_of_swap_gates =num_of_swap_gates;
    this->num_of_qubits = coupling_graph.num_of_qubits;
    this->coupling_graph = coupling_graph;
    this->sample_queue = sample_queue;
    this->res_probs =res_probs;
    this->res_flags = res_flags;
    this->res_values = res_values;
    this->cid = cid;
    long capacity = static_cast<long>(this->num_of_iterations) * static_cast<long>(coupling_graph.num_of_edges) * 50;
    this->mcts_node_pool.resize(capacity);
    this->mode = 1;


}
mcts::RLMCTSTree::~RLMCTSTree(){
    if(mode == 0){
        this->thread_flag->clear();
        this->inferencer.join();
        delete[] this->res_flags, 
               this->res_values, 
               this->res_probs;
         delete this->thread_flag;
    }
}

void mcts::RLMCTSTree::load_data(int num_of_logical_qubits, mcts::Circuit& circuit){
    this->num_of_logical_qubits = num_of_logical_qubits;
    this->circuit = circuit;
}


std::vector<int> mcts::RLMCTSTree::run_rl(){
    mcts::MCTSNode* cur_node = this->root_node, *pre_node = nullptr;
    this->num_of_executed_gates += cur_node->reward;
    for(int i = 0; i < this->num_of_process; i++){
        this->res_flags[i].test_and_set(std::memory_order_relaxed);
    }
    while(!cur_node->is_terminal_node()){
        auto start = std::chrono::system_clock::now();
        if(!is_has_majority(cur_node)){
            if(this->num_of_process > 1){
                std::atomic_int id(0);
                int pid = 0;
                #pragma omp parallel shared(id) private(pid) num_threads(this->num_of_process)
                { 
                    pid = id.fetch_add(1);
                    #pragma omp for
                    for(int i = 0; i < this->num_of_iterations; i++){
                        this->search_thread(cur_node, pid);
                    }
                }
            }else{
                int pid = 0;
                for(int i = 0; i < this->num_of_iterations; i++){
                    this->search_thread(cur_node, pid);
                }
            }
        }
        pre_node = cur_node;
        cur_node = this->decide(cur_node);
        cur_node->parent = nullptr;
        if(cur_node->reward==0){
            this->fallback_count += 1;
        }else{
            this->fallback_count = 0;
        }
        this->num_of_added_swap_gates += 1;
        int best_swap_label = this->coupling_graph.swap_gate_label(cur_node->swap_gate);
        this->swap_label_list.push_back(best_swap_label);

        this->num_of_executed_gates += cur_node->reward;
        if(this->is_generate_data){
            this->generate_data(pre_node, best_swap_label);
        }
        this->delete_tree(pre_node);
        if(this->fallback_count > 10){
            this->fallback(cur_node);
        }
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> cost_time = end - start;
        std::cout<<"------------------------------------------"<<std::endl; 
        std::cout<<"cost time:"<<cost_time.count()<<std::endl;
        std::cout<<"swap gate:   "<<this->num_of_added_swap_gates
                 <<"   executed gate:   "<<this->num_of_executed_gates<<std::endl;
        std::cout<<"------------------------------------------"<<std::endl; 
    }
    std::vector<int> res{this->num_of_executed_gates, this->num_of_added_swap_gates};
    return res;
}
void mcts::RLMCTSTree::search_thread(mcts::MCTSNode* root_node, int pid){
    mcts::MCTSNode* cur_node = this->select(root_node);
    float value = 0;
    if(with_predictor){
        value = this->evaluate(cur_node, pid);
        if(!cur_node->is_terminal_node()){
            this->expand(cur_node);
        }
    }else{
        if(!cur_node->is_terminal_node()){
            this->expand(cur_node);
        } 
        value = this->evaluate(cur_node, pid);
    }
    this->backpropagate(cur_node, value);
}


float mcts::RLMCTSTree::evaluate(mcts::MCTSNode* cur_node, int pid){
    if(cur_node->is_terminal_node()){
        cur_node->value = 0.0;
        return 0.0;
    }
    std::vector<int> subcircuit = cur_node->get_subcircuit(this->threshold_size);
    torch::Tensor adj_tensor = torch::empty({this->threshold_size, 5}, torch::TensorOptions().dtype(torch::kInt32));
    torch::Tensor qubits_tensor = torch::empty({this->threshold_size, 2}, torch::TensorOptions().dtype(torch::kInt32));
    torch::Tensor padding_mask_tensor = torch::ones({this->threshold_size}, torch::TensorOptions().dtype(torch::kBool));
    using namespace torch::indexing;
    padding_mask_tensor.index_put_({Slice(0, None, subcircuit.size())}, 0);

    memcpy(adj_tensor.data_ptr(),
        static_cast<void*>(this->circuit.get_adj_matrix(subcircuit, this->threshold_size).data()),
        sizeof(int)*this->threshold_size*5);

    memcpy(qubits_tensor.data_ptr(),
        static_cast<void*>(this->circuit.get_qubits_matrix(subcircuit, cur_node->qubit_mapping, this->threshold_size).data()),
        sizeof(int)*this->threshold_size*2);

    this->sample_queue->enqueue(std::move(mcts::Sample(this->cid, pid, 
                                         std::move(qubits_tensor), std::move(padding_mask_tensor), std::move(adj_tensor))
                                         ));

    while(this->res_flags[pid].test_and_set(std::memory_order_acquire));
    cur_node->value = this->res_values[pid];
    cur_node->w += this->res_values[pid];
    auto prob_accessor = this->res_probs[pid].accessor<float,1>();
    if(this->extended){
        for(int i = 0; i < cur_node->num_of_children; i++){
            //std::cout<<prob_accessor[i]<<"  ";
            cur_node->probs_of_children[i] = prob_accessor[i];
        }
    }else{
       for(int i = 0; i < cur_node->num_of_children; i++){
            //std::cout<<prob_accessor[i]<<"  ";
            cur_node->probs_of_children[i] = prob_accessor[this->coupling_graph.swap_gate_label(cur_node->candidate_swap_list[i])];
        } 
    }
    return this->res_values[pid];
}


