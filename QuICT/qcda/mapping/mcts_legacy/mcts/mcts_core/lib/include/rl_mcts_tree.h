#ifndef RL_MCTS_TREE
#define RL_MCTS_TREE

#include<string>
#include<torch/torch.h>
#include <torch/script.h>
#include"concurrentqueue.h"
#include"blockingconcurrentqueue.h"
#include"mcts_tree.h"

namespace mcts{
    class Sample{
    public:
        int cid;
        int pid;
        torch::Tensor qubits;
        torch::Tensor padding_mask;
        torch::Tensor adj;
        Sample(){
            this->cid = 0;
            this->pid = 0; 
        }

        Sample(int cid, int pid, 
            torch::Tensor &qubits, torch::Tensor &padding_mask, torch::Tensor &adj){
            this->cid = cid;
            this->pid = pid;
            this->qubits = qubits;
            this->padding_mask = padding_mask;
            this->adj  = adj;
        }

        
        Sample(int cid, int pid, 
            torch::Tensor &&qubits, torch::Tensor &&padding_mask, torch::Tensor &&adj){
            this->cid = cid;
            this->pid = pid;
            this->qubits = std::move(qubits);
            this->padding_mask = std::move(padding_mask);
            this->adj  = std::move(adj);
        }

        Sample(Sample &s){
            this->cid = s.cid;
            this->pid = s.pid;
            this->qubits = s.qubits;
            this->padding_mask = s.padding_mask;
            this->adj  = s.adj;
        }

        
        Sample(Sample &&s){
            this->cid = s.cid;
            this->pid = s.pid;
            this->qubits = std::move(s.qubits);
            this->padding_mask = std::move(s.padding_mask);
            this->adj  = std::move(s.adj);
        }

        Sample& operator=(Sample &s){
            if(this != &s){
                this->cid = s.cid;
                this->pid = s.pid;
                this->qubits = s.qubits;
                this->padding_mask = s.padding_mask;
                this->adj  = s.adj;
            }
            return *this;
        }

        
        Sample& operator=(Sample &&s){
            if(this != &s){
                this->cid = s.cid;
                this->pid = s.pid;
                this->qubits = std::move(s.qubits);
                this->padding_mask = std::move(s.padding_mask);
                this->adj  = std::move(s.adj);
            }
            return *this;
        }
    };

    typedef moodycamel::BlockingConcurrentQueue<Sample> SampleQueue;
    typedef torch::jit::script::Module Model;
    
    void inference_thread(SampleQueue& sample_queue, std::atomic_flag* thread_flag, 
                        int num_of_process, int batch_size, const std::string model_file_path, int device,
                        std::atomic_flag* res_flags, float* res_values, torch::Tensor* res_probs);

    class RLMCTSTree: public MCTSTree{
        public:
            RLMCTSTree(int major, int info, bool extened, bool with_predictor,
                    bool is_generate_data, int threshold_size, 
                    float gamma, float c, float virtual_loss, 
                    int bp_mode, int num_of_process, 
                    int size_of_subcircuits,  int num_of_swap_gates, 
                    int num_of_iterations, int num_of_playout,
                    int num_of_qubits, int feature_dim, int num_of_edges, 
                    int * coupling_graph, int * distance_matrix, 
                    int * edge_label, float * feature_matrix,
                    const char* model_file_path, int device);

            RLMCTSTree(int major, int info, bool extened, bool with_predictor,
                    bool is_generate_data, int threshold_size, 
                    float gamma, float c, float virtual_loss, 
                    int bp_mode, int num_of_process, 
                    int size_of_subcircuits,  int num_of_swap_gates, 
                    int num_of_iterations, int num_of_playout,
                    CouplingGraph& coupling_graph, int cid,
                    mcts::SampleQueue* sample_queue, std::atomic_flag* res_flags,
                    float* res_values, torch::Tensor* res_probs);
            ~RLMCTSTree();
            int mode;
            std::vector<int> run_rl();
            void search_thread(MCTSNode* root_node, int pid);
            float evaluate(MCTSNode* root_node, int pid);

            using mcts::MCTSTree::load_data;
            void load_data(int num_of_logical_qubits, Circuit& circuit);

            
            int cid;
            std::atomic_flag* thread_flag;
            SampleQueue* sample_queue;
            std::atomic_flag* res_flags;
            float* res_values;
            torch::Tensor* res_probs;

            std::thread inferencer;
    }; 
}
#endif