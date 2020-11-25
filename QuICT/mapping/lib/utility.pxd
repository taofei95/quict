cdef extern from "utility.cc":
    pass

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string

cdef extern from "utility.hh" namespace "utility":
    cdef cppclass edge:
        int s
        int t
        int w
        edge() except +
        edge(int source, int terminal, int weight) except +
    
    cdef cppclass qreg:
        string qubitsName
        int numOfQubits
        qreg() except +
        qreg(const string& qubitsNmae, int num) except +
    
    cdef cppclass gate:
        int gateName
        int type
        int ctrl
        int tar
        string p1
        string p2
        string p3
        gate() except +
        gate(int ctrl, int tar, int type, int gateName) except +

    bool fill(vector[int]& mark, int n) except +

    int calDepth(vector[gate]& circuit, int n) except +
    
    int countQubits(vector[gate]& circuit, int n) except +
    
