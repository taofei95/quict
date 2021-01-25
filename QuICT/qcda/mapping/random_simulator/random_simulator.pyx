# distutils: language = c++
# distutils: extra_compile_args = -std=c++11 -fopenmp
# distutils: extra_link_args = -fopenmp
# cython: language_level = 3
# cython: boundscheck = False
# cython: wraparound = False 
import numpy as np
cimport numpy as np
from cython.parallel cimport prange
from libcpp.vector cimport vector
from libcpp.algorithm cimport sort
from libcpp.utility cimport move
from libc.time cimport time


cdef:
    unsigned int  TWO = 2
    int INIT_NUM = -1


cdef extern from "<random>" namespace "std":
    cdef cppclass mt19937 nogil:
        mt19937() # we need to define this constructor to stack allocate classes in Cython
        mt19937(unsigned int seed) # not worrying about matching the exact int type for seed

    cdef cppclass uniform_real_distribution[T] nogil:
        uniform_real_distribution()
        uniform_real_distribution(T a, T b)
        T operator()(mt19937 gen) # ignore the possibility of using other classes for "gen"

cdef float random_generator() nogil:
    cdef:
        mt19937 gen = mt19937(<unsigned int>time(NULL))
        uniform_real_distribution[float] dist = uniform_real_distribution[float](0.0,1.0)
    return dist(gen)

cdef class RandomSimulator:
    cdef:
        int[:,::1] graph, gates, coupling_graph, distance_matrix
        vector[int] front_layer, qubit_mapping, qubit_inverse_mapping, qubit_mask
        int num_of_physical_qubits , num_of_gates, num_of_subcircuit_gates, num_of_iterations, num_of_logical_qubits

    def __cinit__(self, np.ndarray[np.int32_t, ndim = 2] graph, np.ndarray[np.int32_t, ndim = 2] gates,
                np.ndarray[np.int32_t, ndim = 2] coupling_graph, np.ndarray[np.int32_t, ndim = 2] distance_matrix, 
                int num_of_gates, int num_of_logical_qubits):
                self.graph = graph
                self.gates = gates
                self.coupling_graph = coupling_graph
                self.distance_matrix = distance_matrix
                self.num_of_physical_qubits = self.coupling_graph.shape[0]
                self.num_of_logical_qubits = num_of_logical_qubits
                self.num_of_gates = num_of_gates
    
    

    cdef int NNC(self, vector[int] &front_layer, vector[int] qubit_mapping) nogil:
        cdef:
            int i, res = 0
        for i in front_layer:
            res += self.distance_matrix[qubit_mapping[self.gates[i,0]], qubit_mapping[self.gates[i,1]]]
        return res
    
    cdef float _f(self, int x) nogil:
        if x<0:
            return 0.0
        elif x == 0:
            return 0.001
        else:
            return <float>x

    cdef vector[float] f(self, vector[int] &NNC) nogil:
        cdef:
            unsigned int i, n = NNC.size()
            vector[float] res
            float  s = 0
        res.resize(n, INIT_NUM)
        for i in range(n):
            res[i] = self._f(NNC[i])
            s +=  res[i]
        for i in range(n):
            res[i] = res[i] / s
        return res
    
    cdef unsigned int random_choice(self, vector[float] &p) nogil:
        cdef:
            float r = random_generator(), t = 0.0 
            unsigned int i, n = p.size()
        for i in range(n):
            t += p[i]
            if t > r:
                return i
        return n-1 
    
    cdef vector[int] change_qubit_mapping_with_swap_gate(self, vector[int] &qubit_mapping, vector[int] &qubit_inverse_mapping, vector[int] &qubit_mask, 
                                                    vector[int] swap_gate, bint in_palace = False) nogil:
        cdef:
            int qubit_0, qubit_1 ,temp
            vector[int] res_mapping = qubit_mapping
        
        qubit_0 = qubit_inverse_mapping[swap_gate[0]]
        qubit_1 = qubit_inverse_mapping[swap_gate[1]]
        if qubit_0 == -1 and qubit_1 == -1:
            return res_mapping

        if in_palace:
            qubit_inverse_mapping[swap_gate[0]] = qubit_1
            qubit_inverse_mapping[swap_gate[1]] = qubit_0
            
            temp = qubit_mask[swap_gate[0]]
            qubit_mask[swap_gate[0]] = qubit_mask[swap_gate[1]]
            qubit_mask[swap_gate[1]] = temp

            if qubit_0 == -1:
                qubit_mapping[qubit_1] = swap_gate[0]
            elif qubit_1 == -1:
                qubit_mapping[qubit_0] = swap_gate[1]
            else:            
                qubit_mapping[qubit_0] = swap_gate[1]
                qubit_mapping[qubit_1] = swap_gate[0]
        else:
            if qubit_0 == -1:
                res_mapping[qubit_1] = swap_gate[0]
            elif qubit_1 == -1:
                res_mapping[qubit_0] = swap_gate[1]
            else:
                res_mapping[qubit_0] = swap_gate[1]
                res_mapping[qubit_1] = swap_gate[0]
        
        return res_mapping

    cdef vector[int] get_involved_qubits(self, vector[int] &front_layer, vector[int] &qubit_mapping) nogil:
        cdef:
            vector[int] qubits_set
            unsigned int  i, n = front_layer.size()
        qubits_set.reserve(self.num_of_physical_qubits)
        
        for i in range(n):
            qubits_set.push_back(qubit_mapping[self.gates[front_layer[i]][0]])
            qubits_set.push_back(qubit_mapping[self.gates[front_layer[i]][1]])
        
        return qubits_set
        
    cdef vector[int] get_candidate_swap_gate_list(self, vector[int] &front_layer, vector[int] &qubit_mapping) nogil:
        cdef:
            vector[int] qubits_set, qubits_mark
            vector[int] candidate_swap_gate_list
            int i, j
        
        candidate_swap_gate_list.reserve(self.num_of_physical_qubits*10)
        qubits_mark.resize(self.num_of_physical_qubits)
        qubits_set = self.get_involved_qubits(front_layer, qubit_mapping)
        sort(qubits_set.begin(), qubits_set.end())
       
        for i in qubits_set:
            qubits_mark[i] = 1
            for j in range(self.num_of_physical_qubits):
                if qubits_mark[j] == 0 and self.coupling_graph[i,j] == 1:
                    candidate_swap_gate_list.push_back(i)
                    candidate_swap_gate_list.push_back(j)

        return candidate_swap_gate_list

    cdef bint is_executable(self, int index, vector[int] &qubit_mapping) nogil:
        if self.coupling_graph[qubit_mapping[self.gates[index][0]], qubit_mapping[self.gates[index][1]]] == 1:
            return True
        else:
            return False

    cdef bint is_free(self, int index, vector[int] &qubit_mapping, vector[int] &qubit_mask) nogil:
        if (qubit_mask[qubit_mapping[self.gates[index][0]]] == -1 or qubit_mask[qubit_mapping[self.gates[index][0]]] == index) and  (qubit_mask[qubit_mapping[self.gates[index][1]]] == -1 or qubit_mask[qubit_mapping[self.gates[index][1]]] == index):
            return True
        else:
            return False

    cdef int update_front_layer(self, vector[int] &front_layer, vector[int] &qubit_mapping, vector[int] &qubit_mask) nogil:
        cdef:
            vector[int] front_layer_stack
            int num_of_executed_gates = 0, top, suc
            unsigned int i 
        
        front_layer_stack.swap(front_layer)
        while not front_layer_stack.empty():
            top = front_layer_stack.back()
            front_layer_stack.pop_back()
            if self.is_executable(top, qubit_mapping):
                num_of_executed_gates += 1
                for i in range(2):
                    suc = self.graph[top,i]
                    qubit_mask[qubit_mapping[self.gates[top,i]]] = suc 
                    # print(list(self.gates[suc]))
                    # print(qubit_mask)
                    # print(qubit_mapping)
                    if suc != -1:
                        #print(self.is_free(suc, qubit_mapping, qubit_mask))
                        if self.is_free(suc, qubit_mapping, qubit_mask):
                            #print(suc)
                            front_layer_stack.push_back(suc)       
            else:
                front_layer.push_back(top)
            #print(front_layer)
        return num_of_executed_gates

    cdef unsigned int simulation_thread(self) nogil:
        cdef:
            vector[int] front_layer = self.front_layer
            vector[int] qubit_mapping = self.qubit_mapping
            vector[int] qubit_inverse_mapping = self.qubit_inverse_mapping
            vector[int] qubit_mask = self.qubit_mask
            vector[int] candidate_swap_gate_list, NNC, swap_gate
            vector[float] pf
            int NNC_base, j
            unsigned int i, num_of_executed_gates = 0, num_of_candidate_swap_gates, swap_gate_index, num_of_swap_gates = 0

        swap_gate.resize(2, INIT_NUM)
        num_of_executed_gates += self.update_front_layer(front_layer, qubit_mapping, qubit_mask)
        while not front_layer.empty() and num_of_executed_gates < self.num_of_subcircuit_gates:
            candidate_swap_gate_list = self.get_candidate_swap_gate_list(front_layer, qubit_mapping)
            #print(qubit_mapping)
            # for j in front_layer:
            #     print(list(self.gates[j]))
            # print(candidate_swap_gate_list)
            num_of_candidate_swap_gates = <int>(candidate_swap_gate_list.size() / TWO)
            NNC_base = self.NNC(front_layer, qubit_mapping)
            NNC.resize(num_of_candidate_swap_gates, INIT_NUM)
            for i in range(num_of_candidate_swap_gates):
                swap_gate[0] = candidate_swap_gate_list[2*i]
                swap_gate[1] = candidate_swap_gate_list[2*i+1]
               #for j in front_layer:
                #    print(list(self.gates[i])) 
                #print(qubit_mapping)
                #print(self.change_qubit_mapping_with_swap_gate(qubit_mapping, qubit_inverse_mapping, qubit_mask, swap_gate))
                NNC[i] = NNC_base - self.NNC(front_layer, self.change_qubit_mapping_with_swap_gate(qubit_mapping, qubit_inverse_mapping, qubit_mask, swap_gate))
               # print(NNC_base)
                #print(NNC[i])
            pf = self.f(NNC)
            # print(NNC)
            # print(pf)
            swap_gate_index = self.random_choice(pf)
            swap_gate[0] = candidate_swap_gate_list[2*swap_gate_index]
            swap_gate[1] = candidate_swap_gate_list[2*swap_gate_index+1]
            self.change_qubit_mapping_with_swap_gate(qubit_mapping, qubit_inverse_mapping, qubit_mask, swap_gate, True)
            num_of_executed_gates += self.update_front_layer(front_layer, qubit_mapping, qubit_mask)
            num_of_swap_gates += 1 
            #print(num_of_executed_gates)
        return num_of_swap_gates
        
    cpdef int simulate(self, list front_layer, list qubit_mapping, list qubit_mask,
                    int num_of_subcircuit_gates, int num_of_iterations):
            cdef:
                vector[int] res
                unsigned int i , n = self.num_of_logical_qubits
                int  minimum = self.num_of_gates * self.num_of_physical_qubits 
            
            
            self.front_layer = front_layer
            self.qubit_mapping = qubit_mapping
            self.qubit_mask = qubit_mask
            self.qubit_inverse_mapping.resize(self.num_of_physical_qubits, INIT_NUM)

            for i in range(n):
                self.qubit_inverse_mapping[qubit_mapping[i]] = i 

            self.num_of_subcircuit_gates = num_of_subcircuit_gates
            self.num_of_iterations = num_of_iterations
            
            res.resize(self.num_of_iterations)
           
            with nogil:
                for i in prange(self.num_of_iterations):
                    res[i] = self.simulation_thread()
            
            # for i in range(self.num_of_iterations):
            #     res[i] = self.simulation_thread()

            for i in range(self.num_of_iterations):
                if res[i] < minimum:
                    minimum = res[i]

            return minimum
            


            
            

