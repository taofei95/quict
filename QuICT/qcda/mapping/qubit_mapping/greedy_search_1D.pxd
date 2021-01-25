cdef extern from "greedy_search_1D.cc":
    pass
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.pair cimport pair

from utility cimport edge, gate

cdef extern from "greedy_search_1D.hh" namespace "mapping":
    ctypedef vector[gate] Circuit
    # ctypedef bool (*compare)(const int &, const int &)

    # cdef cppclass s_cmp:
    #     bool operator()(const pair[int,float]& first, const pair[int,float]& second)


    # void swap(int& a, int& b)

    # bool L(const int& a, const int &b)
    
    # bool G(const int& a, const int &b)

    # int countInversions(vector[int]& disF, vector[int]& constraint, compare func)
    
    # float byCenter(vector[int]& disF, vector[int]& constraint)

    # float median(vector[int]& disF, vector[int]&  constraint)

    # int countSubConstraintInversion(vector[vector[int]] &constraint, vector[int] &disF, int s, int t)

    # int countConstraintInversion(vector[vector[int]] &constraint, vector[int] &initMapping, vector[int] &permutation)

    # vector[edge] constructGraph(vector[gate] &circuit, int n)
    # int calLA(vector[edge] &graph, vector[int] &permuatation)
    # vector[int] minLA(vector[edge] &graph, int n)
    # vector[int] findInitMapping(vector[gate] &circuit, int n)

    # int findLargestMove(vector[int] &permutation, vector[int] &direction)

    # vector[int] enumerate(vector[int] &initMapping, vector[vector[int]] &constraint)

    # vector[int] searchMapping(vector[int]& initMapping, vector[vector[int]]& constraints)


    # vector[gate] mappingTrans(vector[int]& initMapping, vector[int]& tarMapping)

    # void insertConstraint(vector[int]& constraint, int connect, int hang)

    # void mergeConstraint(vector[int]& constraint1, vector[int]& constraint2, int c1, int c2)


    # vector[vector[int]] findConstraint(vector[gate]& gates, int& pos, int n)

    # vector[gate] logicToPhysics(vector[gate]& gates, vector[int]& mapping, int start, int end)

    Circuit greedySearch(Circuit& circuit, vector[int]& mapping, int n, const string& init_method, const string& search_method) except +
    Circuit globalSifting(Circuit& circuit, vector[int]& mapping, int n) except +