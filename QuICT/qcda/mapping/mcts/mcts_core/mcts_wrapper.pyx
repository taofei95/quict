# distutils: language = c++
# distutils: extra_compile_args="-std=c++14" "-I./lib/include/"
# distutils: extra_link_args="-L./lib/build/"
# cython: language_level = 3
# cython: boundscheck = False
# cython: wraparound = False 



import numpy as np
cimport numpy as np
from cython.operator cimport dereference as deref
from QuICT.qcda.mapping.mcts.mcts_core.mcts_tree cimport MCTSTree
# from mcts_tree cimport RLMCTSTree
# from data_generator cimport DataGenerator

cdef class MCTSTreeWrapper:
    cdef:
        MCTSTree* mcts_cpp
        int[:, ::1] circuit, dependency_graph, coupling_graph, distance_matrix, edge_label
        float[:, ::1] feature_matrix
        int num_of_iterations, num_of_logical_qubits, num_of_playout, num_of_swap_gates
        float gamma, c
        int threshold_size, num_of_edges

    def __cinit__(self, int major, int method, int info, bint extended, bint with_predictor, bint is_generate_data,  int threshold_size, float gamma, float c, float virtual_loss, int bp_mode, int num_of_process, 
                int size_of_subcircuits, int num_of_swap_gates, int num_of_iterations, int num_of_playout, int num_of_edges,  
                np.ndarray[np.int32_t, ndim = 2] coupling_graph, np.ndarray[np.int32_t, ndim = 2] distance_matrix, 
                np.ndarray[np.int32_t, ndim = 2] edge_label, np.ndarray[np.float32_t, ndim = 2] feature_matrix):
       
        self.coupling_graph = coupling_graph
        self.distance_matrix = distance_matrix
        self.feature_matrix = feature_matrix
        self.edge_label = edge_label
        self.num_of_swap_gates = num_of_swap_gates
        self.num_of_edges = num_of_edges
        self.threshold_size = threshold_size
        #print((&self.feature_matrix[0,0]))
        #self.mcts_cpp
        self.mcts_cpp = new MCTSTree(major, method, info, extended, with_predictor, is_generate_data, threshold_size, gamma, c, virtual_loss, bp_mode, num_of_process, 
                                    size_of_subcircuits, num_of_swap_gates, num_of_iterations, num_of_playout, 
                                    <int>self.coupling_graph.shape[0], <int>self.feature_matrix.shape[1], num_of_edges,
                                    &(self.coupling_graph[0,0]), &(self.distance_matrix[0,0]), &(self.edge_label[0][0]), &(self.feature_matrix[0,0]))


    def load_data(self, int num_of_logical_qubits, int num_of_gates, 
                np.ndarray[np.int32_t, ndim = 2] circuit, np.ndarray[np.int32_t, ndim =2] dependency_graph, 
                list qubit_mapping, list qubit_mask, list front_layer):   
        self.circuit = circuit
        self.dependency_graph = dependency_graph
        deref(self.mcts_cpp).load_data(num_of_gates, num_of_logical_qubits, &self.circuit[0][0], &self.dependency_graph[0][0])
        deref(self.mcts_cpp).build_search_tree(qubit_mapping, qubit_mask, front_layer)

    def search(self):
        return deref(self.mcts_cpp).run()

    cpdef get_data(self):
        cdef:
            int num_of_samples = deref(self.mcts_cpp).get_num_of_samples()
            int[:] adj_mem = <int[:num_of_samples*self.threshold_size*5]> deref(self.mcts_cpp).get_adj_list()
            int[:] qubits_mem = <int[:num_of_samples*self.threshold_size*2]> deref(self.mcts_cpp).get_qubits_list() 
            int[:] num_mem = <int[:num_of_samples]> deref(self.mcts_cpp).get_num_list()
            int[:] swap_gate_mem =<int[:num_of_samples]>  deref(self.mcts_cpp).get_swap_label_list()
            float[:] value_mem = <float[:num_of_samples]> deref(self.mcts_cpp).get_value_list()
            float[:] action_prob_mem = <float[:num_of_samples*self.num_of_edges]> deref(self.mcts_cpp).get_action_prob_list()
      
        
        adj = np.asarray(adj_mem)
        adj = adj.reshape((num_of_samples, self.threshold_size, 5))
        
        qubits = np.asarray(qubits_mem)
        qubits = qubits.reshape((num_of_samples, self.threshold_size, 2))

        num = np.asarray(num_mem)
        swap_gate = np.asarray(swap_gate_mem)
        value = np.asarray(value_mem)

        action_prob = np.asarray(action_prob_mem)
        action_prob = action_prob.reshape((num_of_samples, self.num_of_edges))

        return num_of_samples, adj, qubits, num, swap_gate, value, action_prob
    
    def get_swap_gate_list(self):
        return  deref(self.mcts_cpp).get_added_swap_label_list()


    def __dealloc__(self):
        del self.mcts_cpp
    

# cdef class DataGeneratorWrapper:
#     cdef:
#         DataGenerator* data_gen 
#         int[:, ::1] circuit, dependency_graph, coupling_graph, distance_matrix, edge_label
#         float[:, ::1] feature_matrix
#         int[:] adj_mem, qubits_mem, padding_mask_mem, swap_gate_mem 
#         float[:] value_mem, action_prob_mem 

#         int num_of_iterations, num_of_logical_qubits, num_of_playout, num_of_swap_gates
#         float gamma, c
#         int threshold_size, num_of_edges, num_of_qubits, num_of_process
#         char* model_file_path
#         int device, num_of_circuit_process, inference_batch_size, max_batch_size
#         int pool_capcaity, start_threshold

#     def __cinit__(self, int major, int info, bint with_predictor,
#                 bint is_generate_data,  int threshold_size, 
#                 float gamma, float c,
#                 float virtual_loss, int bp_mode, 
#                 int num_of_process, int size_of_subcircuits, 
#                 int num_of_swap_gates, int num_of_iterations, 
#                 int num_of_playout, int num_of_edges, 
#                 np.ndarray[np.int32_t, ndim = 2] coupling_graph, 
#                 np.ndarray[np.int32_t, ndim = 2] distance_matrix, 
#                 np.ndarray[np.int32_t, ndim = 2] edge_label, 
#                 np.ndarray[np.float32_t, ndim = 2] feature_matrix,
#                 bytes model_file_path, int device, 
#                 int num_of_circuit_process, int inference_batch_size, 
#                 int max_batch_size, int pool_capcaity,
#                 int start_threshold, bint extended):
#         feature_dim = feature_matrix.shape[1]
#         self.num_of_qubits = coupling_graph.shape[0]
#         self.num_of_swap_gates = num_of_swap_gates
#         self.num_of_edges = num_of_edges
#         self.threshold_size = threshold_size
#         self.model_file_path = model_file_path
#         self.device = device
#         self.num_of_circuit_process = num_of_circuit_process
#         self.inference_batch_size = inference_batch_size
#         self.max_batch_size = max_batch_size
#         self.num_of_process = num_of_process
#         self.pool_capcaity = pool_capcaity
#         self.start_threshold = start_threshold
        
#         self.coupling_graph = coupling_graph
#         self.distance_matrix = distance_matrix
#         self.feature_matrix = feature_matrix
#         self.edge_label = edge_label



#         data_gen = new DataGenerator(major, info, with_predictor, is_generate_data, threshold_size, 
#                 gamma, c, virtual_loss, bp_mode, num_of_process, 
#                 size_of_subcircuits,  num_of_swap_gates, num_of_iterations, num_of_playout,
#                 self.num_of_qubits, feature_dim, num_of_edges, extended,
#                 &(self.coupling_graph[0][0]), &(self.distance_matrix[0][0]), 
#                 &(self.edge_label[0][0]), &(self.feature_matrix[0][0]),
#                 self.model_file_path, device, num_of_circuit_process, 
#                 inference_batch_size, max_batch_size)

#     def is_enough_samples(self):
#         num_of_samples = deref(self.data_gen).get_num_of_samples()
#         return num_of_samples > self.start_threshold
            

#     def get_batch_sample(self, int batch_size):
        
#         num_of_samples = deref(self.data_gen).get_num_of_samples()

#         if num_of_samples > self.start_threshold:
#             start_point = num_of_samples - self.start_threshold
#             end_point = num_of_samples
#             batch_sample = list(np.random.choice(np.arange(start_point, end_point), batch_size))
            
#             adj_mem = <int[:batch_size*self.threshold_size*5]> deref(self.data_gen).get_adj_list(batch_sample)
#             qubits_mem = <int[:batch_size*self.threshold_size*2]> deref(self.data_gen).get_qubits_list(batch_sample) 
#             num_mem = <int[:batch_size]> deref(self.data_gen).get_num_list(batch_sample)
#             swap_gate_mem =<int[:batch_size]>  deref(self.data_gen).get_swap_label_list(batch_sample)
#             value_mem = <float[:batch_size]> deref(self.data_gen).get_value_list(batch_sample)
#             action_prob_mem = <float[:batch_size*self.num_of_edges]> deref(self.data_gen).get_action_prob_list(batch_sample)

#             adj = np.asarray(adj_mem)
#             adj = adj.reshape((batch_size, self.threshold_size, 5))
        
#             qubits = np.asarray(qubits_mem)
#             qubits = qubits.reshape((batch_size, self.threshold_size, 2))

#             padding_mask = np.zeros(shape = (batch_size, self.threshold_size), dtype = np.bool)
#             for i in range(batch_size):
#                 padding_mask[i, num_mem[i]: ] = 1

#             swap_gate = np.asarray(swap_gate_mem)
#             value = np.asarray(value_mem)

#             action_prob = np.asarray(action_prob_mem)
#             action_prob = action_prob.reshape((batch_size, self.num_of_edges))
            
#             return adj, qubits, padding_mask, swap_gate, value, action_prob

#     def load_circuit(self, int num_of_logical_qubits, int num_of_gates, 
#                 np.ndarray[np.int32_t, ndim = 2] circuit, np.ndarray[np.int32_t, ndim =2] dependency_graph, 
#                 list qubit_mapping, list qubit_mask, list front_layer):
        
#         self.circuit = circuit
#         self.dependency_graph = dependency_graph
#         deref(self.data_gen).load_data(num_of_gates, num_of_logical_qubits,
#                         &(self.circuit[0][0]), &(self.dependency_graph[0][0]), qubit_mapping,
#                         qubit_mask, front_layer)


#     def update_model(self):
#         deref(self.data_gen).update_model()

#     def run(self):
        
#         deref(self.data_gen).run()

#     def clear(self):
        
#         deref(self.data_gen).clear()

#     def __dealloc__(self):
#         del self.data_gen


# cdef class RLMCTSTreeWrapper:
#     cdef:
#         RLMCTSTree* rl_mcts_cpp
#         int[:, ::1] circuit, dependency_graph, coupling_graph, distance_matrix, edge_label
#         float[:, ::1] feature_matrix
#         int num_of_iterations, num_of_logical_qubits, num_of_playout, num_of_swap_gates
#         float gamma, c
#         int threshold_size, num_of_edges
#         char* model_file_path

#     def __cinit__(self, int major, int info, bint extended, bint with_predictor ,bint is_generate_data,  int threshold_size, 
#             float gamma, float c, float virtual_loss,
#             int bp_mode, int num_of_process, 
#             int size_of_subcircuits, int num_of_swap_gates, 
#             int num_of_iterations, int num_of_playout, 
#             int num_of_edges,  
#             np.ndarray[np.int32_t, ndim = 2] coupling_graph, 
#             np.ndarray[np.int32_t, ndim = 2] distance_matrix, 
#             np.ndarray[np.int32_t, ndim = 2] edge_label, 
#             np.ndarray[np.float32_t, ndim = 2] feature_matrix, 
#             bytes model_file_path, int device):
       
#         self.coupling_graph = coupling_graph
#         self.distance_matrix = distance_matrix
#         self.feature_matrix = feature_matrix
#         self.edge_label = edge_label
#         self.num_of_swap_gates = num_of_swap_gates
#         self.num_of_edges = num_of_edges
#         self.threshold_size = threshold_size
#         self.mode_file_path = model_file_path

#         self.rl_mcts_cpp = new RLMCTSTree(major, info, extended, with_predictor, is_generate_data, 
#                                     threshold_size, gamma, c, virtual_loss, 
#                                     bp_mode, num_of_process, 
#                                     size_of_subcircuits, num_of_swap_gates, 
#                                     num_of_iterations, num_of_playout, 
#                                     <int>self.coupling_graph.shape[0], <int>self.feature_matrix.shape[1], num_of_edges,
#                                     &(self.coupling_graph[0,0]), &(self.distance_matrix[0,0]), &(self.edge_label[0][0]), &(self.feature_matrix[0,0]),
#                                     self.model_file_path, device)


#     def load_data(self, int num_of_logical_qubits, int num_of_gates, 
#                 np.ndarray[np.int32_t, ndim = 2] circuit, np.ndarray[np.int32_t, ndim =2] dependency_graph, 
#                 list qubit_mapping, list qubit_mask, list front_layer):   
#         self.circuit = circuit
#         self.dependency_graph = dependency_graph
#         deref(self.rl_mcts_cpp).load_data(num_of_gates, num_of_logical_qubits, &(self.circuit[0][0]), &(self.dependency_graph[0][0]))
#         deref(self.rl_mcts_cpp).build_search_tree(qubit_mapping, qubit_mask, front_layer)
        

#     def search(self):
#         return deref(self.rl_mcts_cpp).run_rl()
    
