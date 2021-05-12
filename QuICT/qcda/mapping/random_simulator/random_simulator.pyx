# distutils: language = c++
# distutils: extra_compile_args = -std=c++14 -fopenmp
# distutils: extra_link_args = -fopenmp
# cython: language_level = 3
# cython: boundscheck = False
# cython: wraparound = False 
import numpy as np
cimport numpy as np
from cython.parallel cimport prange
from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set
from libcpp.algorithm cimport sort
from libc.time cimport time

cimport
numpy as np
import numpy as np
from cython.parallel cimport

prange
from libc.time cimport

time
from libcpp.algorithm cimport

sort
from libcpp.unordered_set cimport

unordered_set
from libcpp.vector cimport

vector


cdef:
    unsigned int  TWO = 2
    int INIT_NUM = -1, ZERO


cdef extern from "<random>" namespace "std":
    cdef cppclass mt19937 nogil:
        mt19937() # we need to define this constructor to stack allocate classes in Cython
        mt19937(unsigned int seed) # not worrying about matching the exact int type for seed

    cdef cppclass uniform_real_distribution[T] nogil:
        uniform_real_distribution()
        uniform_real_distribution(T a, T b)
        T operator()(mt19937 gen) # ignore the possibility of using other classes for "gen"

cdef float random_generator(int seed ) nogil:
    cdef:
        mt19937 gen = mt19937(<unsigned int>time(NULL)*seed)
        uniform_real_distribution[float] dist = uniform_real_distribution[float](0.0,1.0)
    return dist(gen)

cdef class RandomSimulator:
    cdef:
        int[:,::1] graph, gates, coupling_graph, distance_matrix
        vector[int] front_layer, qubit_mapping, qubit_inverse_mapping, qubit_mask
        unordered_set[int] subcircuit
        int num_of_physical_qubits , num_of_gates, num_of_subcircuit_gates, num_of_iterations, num_of_logical_qubits, num_of_swap_gates, mode
        float gamma

    def __cinit__(self, np.ndarray[np.int32_t, ndim = 2] graph, np.ndarray[np.int32_t, ndim = 2] gates,
                np.ndarray[np.int32_t, ndim = 2] coupling_graph, np.ndarray[np.int32_t, ndim = 2] distance_matrix, 
                int num_of_swap_gates, int num_of_gates, int num_of_logical_qubits, float gamma, int mode):
                self.graph = graph
                self.gates = gates
                self.coupling_graph = coupling_graph
                self.distance_matrix = distance_matrix
                self.num_of_physical_qubits = self.coupling_graph.shape[0]
                self.num_of_logical_qubits = num_of_logical_qubits
                self.num_of_gates = num_of_gates
                self.num_of_swap_gates = num_of_swap_gates
                self.gamma = gamma
                self.mode = mode
    
    

    cdef int NNC(self, vector[int] &front_layer, vector[int] qubit_mapping) nogil:
        cdef:
            int i, res = 0
        for i in front_layer:
            res += self.distance_matrix[qubit_mapping[self.gates[i,0]], qubit_mapping[self.gates[i,1]]]
        return res
    
    cdef int NNC_2(self, unordered_set[int]& subcircuit, vector[int] qubit_mapping) nogil:
        cdef:
            int i, res = 0
        for i in subcircuit:
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
    
    cdef unsigned int random_choice(self, vector[float] &p, int seed) nogil:
        cdef:
            float r = random_generator(seed), t = 0.0 
            unsigned int i, n = p.size()
        for i in range(n):
            t += p[i]
            if t > r:
                return i
        return n-1 

    cdef unsigned int argmax(self, vector[int] &v) nogil:
        cdef:
            unsigned int i, n = v.size(), res = 0
            int m = -1

        for i in range(n):
            if v[i] > m:
                m = v[i]
                res = i
        return res

    
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

    cdef bint is_free_2(self, int index, vector[int] &qubit_mask) nogil:
        if (qubit_mask[self.gates[index][0]] == -1 or qubit_mask[self.gates[index][0]] == index) and  (qubit_mask[self.gates[index][1]] == -1 or qubit_mask[self.gates[index][1]] == index):
            return True
        else:
            return False

    cdef int update_front_layer(self, vector[int] &front_layer, vector[int] &qubit_mapping, vector[int] &qubit_mask)nogil except +:
        cdef:
            vector[int] front_layer_stack
            int num_of_executed_gates = 0, top, suc
            unsigned int i 
        
        front_layer_stack.swap(front_layer)
        #print("into")
        while not front_layer_stack.empty():
            top = front_layer_stack.back()
            front_layer_stack.pop_back()
            #print(top)
            #print([self.gates[top][0], self.gates[top][1]])
            if self.is_executable(top, qubit_mapping):
                #print(num_of_executed_gates)
                num_of_executed_gates += 1
                for i in range(2):
                    suc = self.graph[top,i]
                    qubit_mask[qubit_mapping[self.gates[top,i]]] = suc 
                    #print(list(self.gates[suc]))
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


    cdef int update_front_layer_2(self, vector[int] &front_layer, vector[int] &qubit_mapping, vector[int] &qubit_mask, unordered_set[int] &subcircuit) nogil:
        cdef:
            vector[int] front_layer_stack
            int num_of_executed_gates = 0, top, suc
            unsigned int i 
        
        front_layer_stack.swap(front_layer)
        while not front_layer_stack.empty():
            top = front_layer_stack.back()
            front_layer_stack.pop_back()
            if self.is_executable(top, qubit_mapping):
                # if subcircuit.find(top) == subcircuit.end():
                #     print("There is a gate not in the subcircuit.")
                subcircuit.erase(top)
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
    
    
    cdef unordered_set[int] get_subcircuit(self, vector[int] front_layer, vector[int] qubit_mask ,int num_of_subcircuit_gates)nogil:
        cdef:
            unordered_set[int] subcircuit
            int idx, tar, ctrl, i, suc, top

        idx = 0
        while idx < num_of_subcircuit_gates  and (front_layer.size()>0):
            top = front_layer.back()
            front_layer.pop_back()
            subcircuit.insert(top)
            idx += 1
            for i in range(2):
                suc = self.graph[top,i]
                qubit_mask[self.gates[top,i]] = suc 
                if suc != -1:
                    if self.is_free_2(suc, qubit_mask):
                        front_layer.push_back(suc)   
        return subcircuit

    cdef float simulation_thread_swap(self, int seed)nogil except +:
        cdef:
            vector[int] front_layer = self.front_layer
            vector[int] qubit_mapping = self.qubit_mapping
            vector[int] qubit_inverse_mapping = self.qubit_inverse_mapping
            vector[int] qubit_mask = self.qubit_mask 
            vector[int] candidate_swap_gate_list, NNC, swap_gate, logical_qubit_mask
            unordered_set[int] subcircuit = self.subcircuit
            vector[float] res
            vector[float] pf
            int NNC_base, j
            float weight, weighted_executed_gates = 0.0, sim_res, num_of_swap_gates = 0
            unsigned int i, num_of_executed_gates = 0, num_of_candidate_swap_gates, swap_gate_index
        
        res.resize(2, 0)
        swap_gate.resize(2, INIT_NUM)
        #print("ft")
        weighted_executed_gates += <float>(self.update_front_layer(front_layer, qubit_mapping, qubit_mask))
        #num_of_executed_gates += self.update_front_layer(front_layer, qubit_mapping, qubit_mask)
        #print("ft")
        weight = 1.0
        #print(weight)
        while not front_layer.empty() and  num_of_swap_gates < self.num_of_swap_gates:
            candidate_swap_gate_list = self.get_candidate_swap_gate_list(front_layer, qubit_mapping)
            #print(qubit_mapping)
            # for j in front_layer:
            #     print(list(self.gates[j]))
            # print(candidate_swap_gate_list)
            num_of_candidate_swap_gates = <int>(candidate_swap_gate_list.size() / 2)
            NNC_base = self.NNC(front_layer, qubit_mapping)
            NNC.resize(num_of_candidate_swap_gates, INIT_NUM)
            for i in range(num_of_candidate_swap_gates):
                swap_gate[0] = candidate_swap_gate_list[2*i]
                swap_gate[1] = candidate_swap_gate_list[2*i+1]
                # for j in front_layer:
                #     print(list(self.gates[i])) 
                #print(qubit_mapping)
                #print(self.change_qubit_mapping_with_swap_gate(qubit_mapping, qubit_inverse_mapping, qubit_mask, swap_gate))
                NNC[i] = NNC_base - self.NNC(front_layer, self.change_qubit_mapping_with_swap_gate(qubit_mapping, qubit_inverse_mapping, qubit_mask, swap_gate))
               # print(NNC_base)
                #print(NNC[i])
            pf = self.f(NNC)
            # print(NNC)
            # print(pf)
            swap_gate_index = self.random_choice(pf, seed)
            swap_gate[0] = candidate_swap_gate_list[2*swap_gate_index]
            swap_gate[1] = candidate_swap_gate_list[2*swap_gate_index+1]
            self.change_qubit_mapping_with_swap_gate(qubit_mapping, qubit_inverse_mapping, qubit_mask, swap_gate, True)
            #num_of_executed_gates += self.update_front_layer(front_layer, qubit_mapping, qubit_mask)
            weighted_executed_gates += weight * <float>(self.update_front_layer(front_layer, qubit_mapping, qubit_mask))
            weight *= self.gamma
            num_of_swap_gates += 1 
            # print(weight)
            # print(weighted_executed_gates)
            # print(num_of_swap_gates)
           # print(subcircuit)
        
        # res[0] = num_of_swap_gates
        # res[1] = num_of_executed_gates
        #sim_res = num_of_executed_gates * cpow(weight, num_of_swap_gates / 2)
        sim_res = weighted_executed_gates 
        #print(sim_res)
        return sim_res

    cdef int simulation_thread_gates(self, int seed)nogil except +:
        cdef:
            vector[int] front_layer = self.front_layer
            vector[int] qubit_mapping = self.qubit_mapping
            vector[int] qubit_inverse_mapping = self.qubit_inverse_mapping
            vector[int] qubit_mask = self.qubit_mask
            vector[int] candidate_swap_gate_list, NNC, swap_gate, logical_qubit_mask
            unordered_set[int] subcircuit = self.subcircuit
            vector[float] res
            vector[float] pf
            int NNC_base, j, num_of_swap_gates = 0
            float weight, weighted_executed_gates = 0.0, sim_res
            unsigned int i, num_of_executed_gates = 0, num_of_candidate_swap_gates, swap_gate_index
        
        res.resize(2, 0)
        swap_gate.resize(2, INIT_NUM)
        #weighted_executed_gates += <float>(self.update_front_layer(front_layer, qubit_mapping, qubit_mask))
        num_of_executed_gates += self.update_front_layer(front_layer, qubit_mapping, qubit_mask)
        weight = self.gamma
        #print(weight)

        while not front_layer.empty() and  num_of_executed_gates < self.num_of_subcircuit_gates:
            candidate_swap_gate_list = self.get_candidate_swap_gate_list(front_layer, qubit_mapping)
            #print(qubit_mapping)
            # for j in front_layer:
            #     print(list(self.gates[j]))
            # print(candidate_swap_gate_list)
            num_of_candidate_swap_gates = <int>(candidate_swap_gate_list.size() / 2)
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
            swap_gate_index = self.random_choice(pf, seed)
            swap_gate[0] = candidate_swap_gate_list[2*swap_gate_index]
            swap_gate[1] = candidate_swap_gate_list[2*swap_gate_index+1]
            self.change_qubit_mapping_with_swap_gate(qubit_mapping, qubit_inverse_mapping, qubit_mask, swap_gate, True)
            num_of_executed_gates += self.update_front_layer(front_layer, qubit_mapping, qubit_mask)
            #weighted_executed_gates += weight * <float>(self.update_front_layer(front_layer, qubit_mapping, qubit_mask))
            #weight *= self.gamma
            num_of_swap_gates += 1 
            # print(weight)
            # print(weighted_executed_gates)
            # print(num_of_swap_gates)
           # print(subcircuit)
        
        # res[0] = num_of_swap_gates
        # res[1] = num_of_executed_gates
        #sim_res = <float>(num_of_executed_gates) * cpow(weight, num_of_swap_gates / 2)
        #sim_res = weighted_executed_gates
        #print(sim_res)
        return num_of_swap_gates

    cdef vector[int] simulation_thread_extended(self, int seed)nogil except +:
        cdef:
            vector[int] front_layer = self.front_layer
            vector[int] qubit_mapping = self.qubit_mapping
            vector[int] qubit_inverse_mapping = self.qubit_inverse_mapping
            vector[int] qubit_mask = self.qubit_mask 
            vector[int] candidate_swap_gate_list, NNC, swap_gate, logical_qubit_mask
            unordered_set[int] subcircuit = self.subcircuit
            vector[int] res
            vector[float] pf
            int NNC_base, j
            float weight, weighted_executed_gates = 0.0, sim_res
            unsigned int i, num_of_executed_gates = 0, sum_executed_gates = 0 ,num_of_candidate_swap_gates, swap_gate_index,  num_of_swap_gates = 0
        
        res.resize(self.num_of_swap_gates+1, 0)
        swap_gate.resize(2, INIT_NUM)
        #print("ft")
        sum_executed_gates += self.update_front_layer(front_layer, qubit_mapping, qubit_mask)
        #num_of_executed_gates += self.update_front_layer(front_layer, qubit_mapping, qubit_mask)
        weight = 1.0
        #print("ft")
        #print(weight)
        while not front_layer.empty() and  num_of_swap_gates < self.num_of_swap_gates:
            candidate_swap_gate_list = self.get_candidate_swap_gate_list(front_layer, qubit_mapping)
            #print(qubit_mapping)
            # for j in front_layer:
            #     print(list(self.gates[j]))
            # print(candidate_swap_gate_list)
            num_of_candidate_swap_gates = <int>(candidate_swap_gate_list.size() / 2)
            NNC_base = self.NNC(front_layer, qubit_mapping)
            NNC.resize(num_of_candidate_swap_gates, INIT_NUM)
            for i in range(num_of_candidate_swap_gates):
                swap_gate[0] = candidate_swap_gate_list[2*i]
                swap_gate[1] = candidate_swap_gate_list[2*i+1]
                # for j in front_layer:
                #     print(list(self.gates[i])) 
                #print(qubit_mapping)
                #print(self.change_qubit_mapping_with_swap_gate(qubit_mapping, qubit_inverse_mapping, qubit_mask, swap_gate))
                NNC[i] = NNC_base - self.NNC(front_layer, self.change_qubit_mapping_with_swap_gate(qubit_mapping, qubit_inverse_mapping, qubit_mask, swap_gate))
               # print(NNC_base)
                #print(NNC[i])
            pf = self.f(NNC)
            # print(NNC)
            # print(pf)
            swap_gate_index = self.random_choice(pf, seed)
            swap_gate[0] = candidate_swap_gate_list[2*swap_gate_index]
            swap_gate[1] = candidate_swap_gate_list[2*swap_gate_index+1]
            self.change_qubit_mapping_with_swap_gate(qubit_mapping, qubit_inverse_mapping, qubit_mask, swap_gate, True)
            #num_of_executed_gates += self.update_front_layer(front_layer, qubit_mapping, qubit_mask)
            num_of_executed_gates =  self.update_front_layer(front_layer, qubit_mapping, qubit_mask)
            sum_executed_gates +=  num_of_executed_gates
            num_of_swap_gates += 1 
            res[num_of_swap_gates] = num_of_executed_gates
            # print(weight)
            # print(weighted_executed_gates)
            # print(num_of_swap_gates)
           # print(subcircuit)
        
        # res[0] = num_of_swap_gates
        # res[1] = num_of_executed_gates
        #sim_res = num_of_executed_gates * cpow(weight, num_of_swap_gates / 2)
        res[0] = sum_executed_gates
        #print(sim_res)
        return res


    cdef int simulation_thread_determinstic(self) nogil except +:
        cdef:
            vector[int] front_layer = self.front_layer
            vector[int] qubit_mapping = self.qubit_mapping
            vector[int] qubit_inverse_mapping = self.qubit_inverse_mapping
            vector[int] qubit_mask = self.qubit_mask 
            vector[int] candidate_swap_gate_list, NNC, swap_gate, logical_qubit_mask
            unordered_set[int] subcircuit = self.subcircuit
            vector[float] pf
            int NNC_base, j
            float weight, weighted_executed_gates = 0.0, sim_res
            unsigned int i, num_of_executed_gates = 0, sum_executed_gates = 0 ,num_of_candidate_swap_gates, swap_gate_index,  num_of_swap_gates = 0
        
        swap_gate.resize(2, INIT_NUM)
        #print("ft")
        self.update_front_layer(front_layer, qubit_mapping, qubit_mask)
        #num_of_executed_gates += self.update_front_layer(front_layer, qubit_mapping, qubit_mask)
        #print("ft")
        #print(weight)
        while not front_layer.empty():
            candidate_swap_gate_list = self.get_candidate_swap_gate_list(front_layer, qubit_mapping)
            #print(qubit_mapping)
            # for j in front_layer:
            #     print(list(self.gates[j]))
            # print(candidate_swap_gate_list)
            num_of_candidate_swap_gates = <int>(candidate_swap_gate_list.size() / 2)
            NNC_base = self.NNC(front_layer, qubit_mapping)
            NNC.resize(num_of_candidate_swap_gates, INIT_NUM)
            for i in range(num_of_candidate_swap_gates):
                swap_gate[0] = candidate_swap_gate_list[2*i]
                swap_gate[1] = candidate_swap_gate_list[2*i+1]
                # for j in front_layer:
                #     print(list(self.gates[i])) 
                #print(qubit_mapping)
                #print(self.change_qubit_mapping_with_swap_gate(qubit_mapping, qubit_inverse_mapping, qubit_mask, swap_gate))
                NNC[i] = NNC_base - self.NNC(front_layer, self.change_qubit_mapping_with_swap_gate(qubit_mapping, qubit_inverse_mapping, qubit_mask, swap_gate))
               # print(NNC_base)
                #print(NNC[i])
            # print(NNC)
            # print(pf)
            swap_gate_index = self.argmax(NNC)
            swap_gate[0] = candidate_swap_gate_list[2*swap_gate_index]
            swap_gate[1] = candidate_swap_gate_list[2*swap_gate_index+1]
            self.change_qubit_mapping_with_swap_gate(qubit_mapping, qubit_inverse_mapping, qubit_mask, swap_gate, True)
            #num_of_executed_gates += self.update_front_layer(front_layer, qubit_mapping, qubit_mask)
            self.update_front_layer(front_layer, qubit_mapping, qubit_mask)
            num_of_swap_gates += 1 
            # print(weight)
            # print(weighted_executed_gates)
            # print(num_of_swap_gates)
           # print(subcircuit)
        
        # res[0] = num_of_swap_gates
        # res[1] = num_of_executed_gates
        #sim_res = num_of_executed_gates * cpow(weight, num_of_swap_gates / 2)
        #print(sim_res)
        return num_of_swap_gates
        
        
    def  simulate(self, list front_layer, list qubit_mapping, list qubit_mask,
                    int num_of_subcircuit_gates, int num_of_iterations, int simulation_mode):
            cdef:
                vector[int] sim_res, sim_swap
                vector[int] logical_qubit_mask, res_int
                vector[float] res
                vector[vector[int]] extended_res
                unsigned int i , n
                int num_of_executed_gates, idx, maximum_int = -1
                int  minimum = self.num_of_gates * self.num_of_physical_qubits
                float average = 0, maximum = -1
            
            
            self.front_layer = front_layer
            
            self.qubit_mapping = qubit_mapping
            self.qubit_mask = qubit_mask
            self.qubit_inverse_mapping.resize(self.num_of_physical_qubits, INIT_NUM)
            
            n = self.qubit_mapping.size()
            # print(front_layer)
            # print(qubit_mapping)
            # print(qubit_mask)
            # for g in front_layer:
            #     print(self.graph[g])
            #     print(self.gates[g])
            
            for i in range(n):
                self.qubit_inverse_mapping[qubit_mapping[i]] = i 
   
            if num_of_subcircuit_gates == -1:
                self.num_of_subcircuit_gates = self.num_of_gates
            else:
                self.num_of_subcircuit_gates = min(num_of_subcircuit_gates, self.num_of_gates)
            
            self.num_of_iterations = num_of_iterations
            
            sim_res.resize(self.num_of_iterations)
            sim_swap.resize(self.num_of_iterations)
            res.resize(self.num_of_iterations)
            res_int.resize(self.num_of_iterations)
            if simulation_mode == 2:
                extended_res = vector[vector[int]](self.num_of_iterations, vector[int](self.num_of_swap_gates, 0))


            # logical_qubit_mask.resize(self.num_of_logical_qubits, -1)
            # for i in range(self.num_of_logical_qubits):
            #     logical_qubit_mask[self.qubit_inverse_mapping[i]] = self.qubit_mask[i]

            # self.subcircuit = self.get_subcircuit(front_layer, logical_qubit_mask, self.num_of_subcircuit_gates)
            #print(self.subcircuit)
            #print("it")
            if simulation_mode == 3:
                return self.simulation_thread_determinstic()

            with nogil:
                for i in prange(self.num_of_iterations):
                    if  simulation_mode == 0:
                        res[i] = self.simulation_thread_swap(i)
                    elif simulation_mode == 1:
                        res_int[i] = self.simulation_thread_gates(i)
                    elif simulation_mode == 2:
                        extended_res[i] = self.simulation_thread_extended(i)

            #print(res[i])
            # print("success")        

            # for i in range(self.num_of_iterations):
            #     res[i] = self.simulation_thread()

            # for i in range(self.num_of_iterations):
            #     if sim_res[i] < minimum:
            #         minimum = sim_res[i]
            #         num_of_executed_gates = sim_swap[i]
            if self.mode == 0:
                for i in range(self.num_of_iterations):
                    average += res[i]
                average = average /  self.num_of_iterations
                return average
            elif self.mode == 1:
                if simulation_mode == 0:
                    for i in range(self.num_of_iterations):
                        maximum  = max(res[i], maximum)
                    #print(maximum)
                    return maximum
                elif simulation_mode == 1:
                    for i in range(self.num_of_iterations):
                        minimum  = min(res_int[i], minimum)
                    #print(maximum)
                    return minimum
                elif simulation_mode == 2:
                    for i in range(self.num_of_iterations):
                        if extended_res[i][0] > maximum_int:
                            maximum_int = extended_res[i][0]
                            idx = i
                    return extended_res[idx]                  
            else:
                return -1
            #print("%d and %d"%(minimum, num_of_executed_gates))
            
            


            
            

