#distutils: language = c++ 
#cython: language_level=3

from libcpp.vector cimport vector
from utility cimport  gate

from greedy_search_1D cimport Circuit, greedySearch, globalSifting

cdef class qubitMapping:
    cdef Circuit c_circuit 
    cdef vector[int] initMapping 
    cdef int n
    def __cinit__(self, circuit,  num,  method = "greedy_search"):
        self.n = num 
        self.initMapping = vector[int](num,0)
        for i in range(num):
            self.initMapping[i] = i
         
        cdef Circuit t_circuit
        cdef gate temp
        for g  in  circuit:
            temp = gate(g['ctrl'], g['tar'], g['type'],g['name'])
            #print("%d %d "%(g['ctrl'], g['tar']))
            t_circuit.push_back(temp)
        # cdef gate gt
        # for gt in t_circuit:
        #     print("%d %d "%(gt.ctrl, gt.tar))
        # cdef int elm 
        # for elm in self.initMapping:
        #     print("%d"%(elm))
        # print(self.n)
        if method == "greedy_search":
            self.c_circuit = greedySearch(t_circuit, self.initMapping, self.n)
        elif method == "global_sifting":
            self.c_circuit = globalSifting(t_circuit, self.initMapping, self.n)
    

    def get_circuit(self):
        init = []
        circuit = []
        num = self.n
        cdef gate g
        cdef int i 
        for i in self.initMapping:
            init.append(i)

        for g in self.c_circuit:
            temp = {}
            temp['ctrl'] = g.ctrl
            temp['tar'] = g.tar
            temp['type'] = g.type
            temp['name'] = g.gateName
            circuit.append(temp)
            
        return (circuit, init)

    def print_circuit(self):
        cdef gate g
        cdef int i 
        for i in self.initMapping:
            print("%d " %(i),end = "")
        print(" ")
       
        for g in self.c_circuit:
            print("%d   %d " %(g.ctrl, g.tar))

            



