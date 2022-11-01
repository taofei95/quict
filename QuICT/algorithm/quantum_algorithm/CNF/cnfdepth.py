# -*- coding:utf8 -*-
# @TIME    : 2022/7/
# @Author  : Cheng Guo
# @File    : 
#from builtins import print
import math
import numpy as np
from QuICT.core import *     #Circuit
from QuICT.core.gate import *
from QuICT.qcda.synthesis.mct import MCTOneAux
from QuICT.qcda.synthesis.mct import MCTLinearHalfDirtyAux
from QuICT.core.operator import Trigger
import logging
from typing import List, Tuple
# from QuICT.qcda.synthesis.mct.mct_linear_simulation import half_dirty_aux
from QuICT.qcda.optimization.commutative_optimization import *

#from .._synthesis import Synthesis
#import random import logging from math import pi, gcd
#from fractions import Fraction
#pyfrom sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from QuICT.core import Circuit
#from QuICT.core.gate import *cc
# from QuICT.simulation.cpu_simulator import CircuitSimulator
#from QuICT.qcda.optimization._optimization import Optimization

class CNFSATDEPTHOracle:
     
    def __init__(self, simu = None):
        self.simulator = simu

    def circuit(self) -> CompositeGate:
        return self._cgate
    
    def run(
        self,
        cnf_file: str,
        ancilla_qubits_num: int = 9 #,
        #qubitnumber: int = 29
    ) -> int:
        """ Run CNF algorithm

        Args:
            cnf_file (str): The file path
            ancilla_qubits_num (int): >= 3
        """
        # check if Aux > 8
        assert ancilla_qubits_num > 8 , "Need at least 9 auxiliary qubit."
        #assert qubitnumber > 29 , "Need at least 9 auxiliary qubit."

        # Step 1: Read CNF File
        variable_number, clause_number, clause_length, CNF_data = self.read_CNF(cnf_file)

        # Step 2: Construct Circuit
        self._cgate = CompositeGate()
        S = math.floor( clause_length / math.log( ancilla_qubits_num, 2 ))
        if S < 1:
            S = 1
        CleanQubitNumber = math.floor((ancilla_qubits_num) / (S+1))   #干净、脏比特数目   其+1 = 非底层能处理的块数的2 倍
        assert CleanQubitNumber > 2 , "Need more auxiliary qubit for enough momery space."
        MemoryQubitNumber = ancilla_qubits_num - 2 * CleanQubitNumber #暂存比特数目 记录
        assert ancilla_qubits_num > 2 * clause_length + 2 , "Need more auxiliary qubit or some CNF-clause contains too many variables.."
        ClauseForFirstDep = 1 + math.floor( MemoryQubitNumber / clause_length) #最底层子句数目
        p = math.floor((CleanQubitNumber + 1)/2)
        depth = math.ceil(math.log( math.ceil(clause_number / ClauseForFirstDep), p )) + 1
        target = variable_number
        if clause_number == 1:
            #n= variable_number + Aux + 1
            controls = CNF_data[1]
            controls_abs=[]
            controls_X=[]
            for i in range(len(controls)):
                if controls[i] < 0:
                    controls_abs.append(-controls[i]-1)
                if controls[i] > 0:
                    controls_abs.append(controls[i]-1)
                    controls_X.append(controls[i]-1)
            for i in range(len(controls_X)):
                X | self._cgate(controls_X[i])
            X | self._cgate(target)
            
            d = set(range(variable_number + 1 + ancilla_qubits_num))
            d.remove(variable_number)
            for j in controls_abs:
                d.remove(j)
            d = list(d)                 

            if controls_abs != []:
                MCTLinearHalfDirtyAux().execute( len(controls_abs) , (1 + len(controls_abs) + len(d))) | self._cgate(controls_abs + d + [target] )

            #if controls_abs != []:
            #    MCTOneAux().execute(len(controls_abs) + 2) | self._cgate(controls_abs + [ target , current_Aux]) 
            # one_dirty_aux(self._cgate, controls_abs, target, current_Aux) #QuICT.qcda.synthesis.mct.
            #X | self._cgate(target)
            for i in range(len(controls_X)):
                X | self._cgate(controls_X[i])
        else:
            
            if ((clause_number - 1 ) * (clause_length + 1 ) + 2 * clause_number +1 ) < ancilla_qubits_num :
                c=[]
                for i in range(variable_number, variable_number + ancilla_qubits_num +1):
                    if (i != target):
                        c.append(i)

                for j in range(2, clause_number + 1, 1):
                    controls = CNF_data[j]
                    for jj in range(len(controls)):
                        CX | self._cgate([abs(controls[jj])-1, (j - 1) * clause_length  + jj + variable_number + 2 * clause_number +1])

                controls = CNF_data[1]
                controls_abs0=[]
                controls_X0=[]
                for i in range(len(controls)):
                    if controls[i] < 0:
                        controls_abs0.append(-controls[i]-1)
                    if controls[i] > 0:
                        controls_abs0.append(controls[i]-1)
                        controls_X0.append(controls[i]-1)
                for i in range(len(controls_X0)):
                    X | self._cgate(controls_X0[i])
                X | self._cgate(c[0])

                if controls_abs0 != []:
                    MCTOneAux().execute(len(controls_abs0) + 2) | self._cgate(controls_abs0 + [ c[0] , c[0] + clause_number ])
                # one_dirty_aux(self._cgate, controls_abs, target, current_Aux)
                #X | self._cgate(target)
                for i in range(len(controls_X0)):
                    X | self._cgate(controls_X0[i])

                for j in range(2, clause_number + 1):
                    controls = CNF_data[j]
                    controls_abs=[]
                    controls_X=[]
                    for jj in range(len(controls)):
                        if controls[jj] < 0:
                            controls_abs.append((j - 1) * clause_length + jj + variable_number + 2 * clause_number +1)
                        if controls[jj] > 0:
                            controls_abs.append((j - 1) * clause_length + jj + variable_number + 2 * clause_number +1)
                            controls_X.append((j - 1) * clause_length + jj + variable_number + 2 * clause_number +1)
                    for i in range(len(controls_X)):
                        X | self._cgate(controls_X[i])
                    X | self._cgate(c[j - 1])
                    if controls_abs != []:
                        MCTOneAux().execute(len(controls_abs) + 2) | self._cgate(controls_abs + [ c[j-1]  , c[j-1] + clause_number ])
                    # one_dirty_aux(self._cgate, controls_abs, target, current_Aux)
                    #X | self._cgate(target)
                    for i in range(len(controls_X)):
                        X | self._cgate(controls_X[i])
                a = 0                  
                if clause_number == 2:
                    CCX | self._cgate([c[0], c[1], target])
                else:
                    b = clause_number
                    for j in range(clause_number-1):
                        CCX | self._cgate([c[a + 2*j], c[a + 2*j + 1], c[b+j]])
                        
                    CCX | self._cgate([c[b + clause_number - 2], c[b + clause_number - 3 ], target])
                
                #还原    
                    for j in range(clause_number-1):
                        CCX | self._cgate([c[a + 2*j], c[a + 2*j + 1], c[b+j]])
                        
                for j in range(2, clause_number + 1, 1):
                    controls = CNF_data[j]
                    for jj in range(len(controls)):
                        CX | self._cgate([abs(controls[jj])-1, (j - 1) * clause_length  + jj + variable_number + 2 * clause_number +1])

                controls = CNF_data[1]
                controls_abs0=[]
                controls_X0=[]
                for i in range(len(controls)):
                    if controls[i] < 0:
                        controls_abs0.append(-controls[i]-1)
                    if controls[i] > 0:
                        controls_abs0.append(controls[i]-1)
                        controls_X0.append(controls[i]-1)
                for i in range(len(controls_X0)):
                    X | self._cgate(controls_X0[i])
                X | self._cgate(c[0])

                if controls_abs0 != []:
                    MCTOneAux().execute(len(controls_abs0) + 2) | self._cgate(controls_abs0 + [ c[0] , c[0] + clause_number  ])
                # one_dirty_aux(self._cgate, controls_abs, target, current_Aux)
                #X | self._cgate(target)
                for i in range(len(controls_X0)):
                    X | self._cgate(controls_X0[i])

                for j in range(1 + 1, clause_number + 1):
                    controls = CNF_data[j]
                    controls_abs=[]
                    controls_X=[]
                    for jj in range(len(controls)):
                        if controls[jj] < 0:
                            controls_abs.append((j - 1) * clause_length   + jj + variable_number + 2 * clause_number +1)
                        if controls[jj] > 0:
                            controls_abs.append((j - 1) * clause_length + jj + variable_number + 2 * clause_number +1)
                            controls_X.append((j - 1) * clause_length + jj + variable_number + 2 * clause_number +1)
                    for i in range(len(controls_X)):
                        X | self._cgate(controls_X[i])
                    X | self._cgate(c[j - 1])

                    if controls_abs != []:
                        MCTOneAux().execute(len(controls_abs) + 2) | self._cgate(controls_abs + [ c[j-1]  , c[j-1] + clause_number ])
                    # one_dirty_aux(self._cgate, controls_abs, target, current_Aux)
                    #X | self._cgate(target)
                    for i in range(len(controls_X)):
                        X | self._cgate(controls_X[i])
            else:

                if (clause_number < 1 + math.floor( (CleanQubitNumber + 1) /2)): #有空 可以修修
                    controls_XX=[]
                    for j in range(1, clause_number + 1):
                        controls = CNF_data[j]
                        controls_abs=[]
                        controls_X=[]
                        for jj in range(len(controls)):
                            if controls[jj] < 0:
                                controls_abs.append( - controls[jj] -1)
                            if controls[jj] > 0:
                                controls_abs.append( controls[jj]- 1)
                                controls_X.append( controls[jj]-1)
                        for i in range(len(controls_X)):
                            X | self._cgate(controls_X[i])
                        X | self._cgate(variable_number + CleanQubitNumber + j )

                        controls_XX.append(variable_number + CleanQubitNumber + j)
                        
                        d = set(range(variable_number + 1 + ancilla_qubits_num))
                        d.remove(variable_number + CleanQubitNumber + j)
                        for jj in controls_abs:
                            d.remove(jj)
                        d = list(d)                           
                        if controls_abs != []:
                            #MCTLinearHalfDirtyAux().execute(len(controls_abs), 1 + variable_number + ancilla_qubits_num) | self._cgate(controls_abs + d + [j + CleanQubitNumber + variable_number] )
                            MCTOneAux().execute(len(controls_abs) + 2) | self._cgate(controls_abs + [ variable_number + CleanQubitNumber + j  , variable_number + j ])
                        else:
                            X | self._cgate(variable_number + CleanQubitNumber + j)
                        # one_dirty_aux(self._cgate, controls_abs, target, current_Aux)
                        #X | self._cgate(target)
                        for i in range(len(controls_X)):
                            X | self._cgate(controls_X[i])
                    
                    
                    d = set(range(variable_number + 1 + ancilla_qubits_num))
                    d.remove(variable_number)
                    for j in controls_XX:
                        d.remove(j)
                    d = list(d)  
                    if controls_XX != []:
                        MCTLinearHalfDirtyAux().execute(len(controls_XX), 1 + variable_number + ancilla_qubits_num) | self._cgate(controls_XX + d + [variable_number] )
                           
                        #MCTOneAux().execute(len(controls_XX) + 2) | self._cgate(controls_XX + [ variable_number , variable_number - 1 ])
                    else:
                        X | self._cgate(variable_number)

                    for j in range(1, clause_number + 1):
                        controls = CNF_data[j]
                        controls_abs=[]
                        controls_X=[]
                        for jj in range(len(controls)):
                            if controls[jj] < 0:
                                controls_abs.append( - controls[jj] -1)
                            if controls[jj] > 0:
                                controls_abs.append( controls[jj]- 1)
                                controls_X.append( controls[jj]-1)
                        for i in range(len(controls_X)):
                            X | self._cgate(controls_X[i])
                        X | self._cgate(variable_number + CleanQubitNumber + j )
                        
                        d = set(range(variable_number + 1 + ancilla_qubits_num))
                        d.remove(variable_number + CleanQubitNumber + j)
                        for jj in controls_abs:
                            d.remove(jj)
                        d = list(d)                           
                        if controls_abs != []:
                            #MCTLinearHalfDirtyAux().execute(len(controls_abs), 1 + variable_number + ancilla_qubits_num) | self._cgate(controls_abs + d + [j + CleanQubitNumber + variable_number] )
                            MCTOneAux().execute(len(controls_abs) + 2) | self._cgate(controls_abs + [ variable_number + CleanQubitNumber + j  , variable_number + j ])
                        else:
                            X | self._cgate(variable_number + CleanQubitNumber + j)
                        # one_dirty_aux(self._cgate, controls_abs, target, current_Aux)
                        #X | self._cgate(target)
                        for i in range(len(controls_X)):
                            X | self._cgate(controls_X[i])                       
                else:
                    p = math.floor( (CleanQubitNumber + 1) /2)                   
                    depth = math.ceil(math.log( clause_number , p)) - 1
                    #print(depth , p , variable_number , CleanQubitNumber)
                    if depth < 1:
                        depth = 0
                    block_len = p ** depth
                    block_number = math.ceil(clause_number / block_len )
                    
                    controls = []
                    for j in range(block_number):
                        self.clause(
                            CNF_data, variable_number, ancilla_qubits_num, clause_length, CleanQubitNumber,
                            j * block_len +1, np.minimum( (j+1) * block_len, clause_number),
                            variable_number + CleanQubitNumber - p + 1 + j, depth-1, depth
                        )
                        controls.append(variable_number + CleanQubitNumber - p + 1 + j)

                    d = set(range(variable_number + 1 + ancilla_qubits_num))
                    d.remove(variable_number)
                    for j in controls:
                        d.remove(j)
                    d = list(d)                 

                    if controls != []:
                        MCTLinearHalfDirtyAux().execute( len(controls) , (1 + len(controls)+len(d))) | self._cgate(controls + d + [ variable_number] )
                            
                    #    MCTOneAux().execute(len(controls) + 2) | self._cgate(controls + [ target, current_Aux] ) 
                    # one_dirty_aux(self._cgate, controls, target, current_Aux)
                    
                    for j in range(block_number):
                        self.clause(
                            CNF_data, variable_number, ancilla_qubits_num, clause_length, CleanQubitNumber,
                            j * block_len +1, np.minimum( (j+1) * block_len, clause_number),
                            variable_number + CleanQubitNumber - p + 1 + j, depth-1, depth
                        )
                



    def read_CNF(self, cnf_file):
        # file analysis
        #输入一个文件，输出CNF格式的list
        variable_number = 0
        clause_number = 0
        CNF_data = []
        clause_length = 0
        f = open(cnf_file, 'r') 
        for line in f.readlines():
            new = line.strip().split()
            int_new=[]
            if new[0] == 'p':
                variable_number = int(new[2])
                clause_number = int(new[3])
            else:
                for x in new:
                    if (x != '0') and (int(x) not in int_new):
                        int_new.append(int(x))
                        if (- int(x)) in int_new:
                            int_new = []
                            break
            CNF_data.append(int_new)  #给各个Clause 编号0,1 ...m-1#
            if len(int_new) > clause_length :
                clause_length = len(int_new)
    
        f.close()

        return variable_number, clause_number, clause_length, CNF_data

    def clause(self, CNF_data: List, variable_number: int, Aux: int, clause_length :int, CleanQubitNumber  :int, StartID: int, EndID: int, target: int, current_depth: int, depth: int):

        p = math.floor((CleanQubitNumber + 1)/2) 
        if StartID == EndID:             #n= variable_number + Aux + 1
            controls = CNF_data[StartID]
            controls_abs=[]
            controls_X=[]
            for i in range(len(controls)):
                if controls[i] < 0:
                    controls_abs.append(-controls[i]-1)
                if controls[i] > 0:
                    controls_abs.append(controls[i]-1)
                    controls_X.append(controls[i]-1)
            for i in range(len(controls_X)):
                X | self._cgate(controls_X[i])

            
            d = set(list(range(variable_number + 1 + Aux)))
            d.remove(target)
            for j in controls:
                if j in d:
                    d.remove(j)
            d = list(d)                 
            X | self._cgate(target)
            if controls_abs != []:
                MCTLinearHalfDirtyAux().execute( len(controls_abs) , (len(controls_abs)+1 + len(d))) | self._cgate(controls_abs + d + [target] )

            #current_Aux  = target + CleanQubitNumber
            #MCTOneAux().execute(len(controls_abs) + 2) | self._cgate(controls_abs + [ target, current_Aux ])
            # one_dirty_aux(self._cgate, controls_abs, target, current_Aux)
            #X | self._cgate(target)
            for i in range(len(controls_X)):
                X | self._cgate(controls_X[i])
        else:  #StartID 和 EndID 不同 
            #if ((EndID - StartID) * (clause_length + 1) < (Aux -  CleanQubitNumber + 1)) and ((EndID - StartID ) < CleanQubitNumber) : #if block_number == 1 而且 EndID - StartID > 1 
            if EndID - StartID  < p :
                
                c=[]
                for i in range(variable_number , variable_number + CleanQubitNumber + 1):
                    if (i != target):
                        c.append(i)

                Parallel_depth_list = [] 
                for j in range(EndID - StartID + 1):
                    Parallel_depth_list.append([])
                variable_check_list = []
                variable_Parallel_value = [ 0 ] * (variable_number)
                clause_Parallel_value = [ 1 ] * (EndID - StartID + 1)
                max_Parallel_depth = 1
                for i in range(StartID, EndID + 1):
                    for j in range(len(CNF_data[i])):
                        #if ((abs(CNF_data[i][j])-1) not in variable_check_list):
                            #variable_check_list.append((abs(CNF_data[i][j])-1))
                            #print(i , j, abs(CNF_data[i][j])-1, len(variable_Parallel_value))
                        variable_Parallel_value[abs(CNF_data[i][j])-1] += 1
                        #else:
                        #    variable_Parallel_value[abs(CNF_data[i][j])-1] += 1
                        if variable_Parallel_value[abs(CNF_data[i][j])-1] > clause_Parallel_value[i-StartID]: 
                            clause_Parallel_value[i-StartID] = variable_Parallel_value[abs(CNF_data[i][j])-1]
                    if clause_Parallel_value[i-StartID] > max_Parallel_depth:
                        max_Parallel_depth = clause_Parallel_value[i-StartID]
                    Parallel_depth_list[(clause_Parallel_value[i-StartID]-1)].append(i)
                
                qmemo = Aux - 2 * CleanQubitNumber #memory qubit number
                Parallel_depth_max = len(Parallel_depth_list[0])
                for k in range(max_Parallel_depth):
                    if Parallel_depth_max < len(Parallel_depth_list[k]):
                            Parallel_depth_max = len(Parallel_depth_list[k])
                #print(Parallel_depth_list)        
                if  (( 1 + clause_length * 2 ) * (Parallel_depth_max + qmemo)) > variable_number +  qmemo: #检验 mct 辅助位是否足够
                    #辅助位不足  
                    #print("buzu")
                    cl_postition = 0
                    for k in range(max_Parallel_depth):                   
                        for kk in range(len(Parallel_depth_list[k])):
                            clasueID = Parallel_depth_list[k][kk]  
                            controls = CNF_data[clasueID]
                            controls_abs0=[]
                            controls_X0=[]
                            for i in range(len(controls)):
                                if controls[i] < 0:
                                    controls_abs0.append(-controls[i]-1)
                                if controls[i] > 0:
                                    controls_abs0.append(controls[i]-1)
                                    controls_X0.append(controls[i]-1)
                            for i in range(len(controls_X0)):
                                X | self._cgate(controls_X0[i])
                            X | self._cgate(cl_postition + variable_number + CleanQubitNumber + 1 )
                            if controls_abs0 != []:
                                #MCTLinearHalfDirtyAux().execute(len(controls_abs0), variable_number + Aux +1) | self._cgate(controls_abs0 + d + [ c[0] ])
                                MCTOneAux().execute(len(controls_abs0) + 2) | self._cgate( controls_abs0 + [  cl_postition + variable_number + CleanQubitNumber + 1 , c[cl_postition] ] )
                            # one_dirty_aux(self._cgate, controls_abs0, target, current_Aux)
                            #X | self._cgate(target)
                            for i in range(len(controls_X0)):
                                X | self._cgate(controls_X0[i])
                            cl_postition += 1

                    b = cl_postition
                    if EndID - StartID == 1:
                        CCX | self._cgate([variable_number + 1 + CleanQubitNumber, variable_number + 1 + 1 + CleanQubitNumber, target])
                    else:
                        for j in range(b-1):
                            CCX | self._cgate([variable_number + 1 + 2 * j + CleanQubitNumber , 
                                                variable_number + 2 + 2 * j + CleanQubitNumber, 
                                                variable_number + 1 + cl_postition + CleanQubitNumber])
                            cl_postition = cl_postition + 1

                        CCX | self._cgate([variable_number - 1 + cl_postition + CleanQubitNumber, variable_number + cl_postition + CleanQubitNumber, target])
                        
                        for j in range(b-2, -1 , -1):
                            CCX | self._cgate([variable_number + 1 + 2 * j + CleanQubitNumber , 
                                                variable_number + 2 + 2 * j + CleanQubitNumber, 
                                                variable_number  + cl_postition + CleanQubitNumber])
                            cl_postition = cl_postition - 1
                    #还原

                    cl_postition = 0
                    for k in range(max_Parallel_depth):
                        for kk in range(len(Parallel_depth_list[k])):
                            clasueID = Parallel_depth_list[k][kk]  
                            controls = CNF_data[clasueID]
                            controls_abs0 = []
                            controls_X0   = []
                            for i in range(len(controls)):
                                if controls[i] < 0:
                                    controls_abs0.append(-controls[i]-1)
                                if controls[i] > 0:
                                    controls_abs0.append(controls[i]-1)
                                    controls_X0.append(controls[i]-1)
                            for i in range(len(controls_X0)):
                                X | self._cgate(controls_X0[i])
                            X | self._cgate(cl_postition + variable_number + CleanQubitNumber + 1 )
                            if controls_abs0 != []:
                                #MCTLinearHalfDirtyAux().execute(len(controls_abs0), variable_number + Aux +1) | self._cgate(controls_abs0 + d + [ c[0] ])
                                #print([  cl_postition + variable_number + CleanQubitNumber + 1 , c[cl_postition] ],controls_abs0)
                                MCTOneAux().execute(len(controls_abs0) + 2) | self._cgate( controls_abs0 + [  cl_postition + variable_number + CleanQubitNumber + 1 , c[cl_postition] ] )                            # one_dirty_aux(self._cgate, controls_abs0, target, current_Aux)

                                #MCTOneAux().execute(len(controls_abs0) + 2) | self._cgate( controls_abs0 + [  cl_postition + variable_number + CleanQubitNumber + 1 , c[cl_postition] ] )
                                              #X | self._cgate(target)
                            for i in range(len(controls_X0)):
                                X | self._cgate(controls_X0[i])
                            cl_postition += 1

                else: #辅助位充足:                     # 
                    # 利用CX 门将合适的 qubit  分配一下
                    # 安排辅助比特位置
                    #print("zu")
                    Parallel_depth_list = [] 
                    for j in range(EndID - StartID + 1):
                        Parallel_depth_list.append([])

                    variable_Parallel_value_max = variable_Parallel_value[0]
                    for j in range(variable_number):
                        if variable_Parallel_value_max < variable_Parallel_value[j]: 
                            variable_Parallel_value_max = variable_Parallel_value[j]
                    
                    tt = variable_Parallel_value_max #tt 运行如下while 之后就是每次并行最大子句数 最后tt返回底层最大并行的层数
                    tb = 1
                    ans = tt
                    #print("variable_Parallel_value_max", variable_Parallel_value_max)                 
                    while ( tb < tt  ): # 最小化最大值 分配辅助位
                        mid = math.floor((tb+tt)/2)
                        if mid == 0:
                            break
                        t = 0
                        for k in range(variable_number):
                            t += math.ceil((variable_Parallel_value[j]) / mid ) - 1 
                        if t  <  qmemo + 1 : #OK
                            ans = mid
                            tt = mid - 1  
                        else:
                            tb = mid + 1 
                    #print(tt , ans , "qmemo", qmemo)
                    #mappinglise 记录新的mapping
                    tt = ans

                    mappingList = []   #存储 CX 门位置
                    mapping_variable =[] #存储哪些变量将映射了至其他位置
                    ta = 0 #ta 在for循环后，记录了多少qmemo 被使用了
                    tb = 0
                    for j in range(variable_number):
                        if variable_Parallel_value[j] > tt:
                            mapping_variable.append(j)
                            for jj in range(math.ceil( variable_Parallel_value[j] / tt)):
                                mappingList.append( [j, variable_number + 1 + 2 * CleanQubitNumber + ta + jj] )
                                
                            ta += math.ceil( variable_Parallel_value[j] / tt - 1)
                    #print("ta",ta)
                    for j in range(ta):
                        CX | self._cgate([ mappingList[j][0], mappingList[j][1]] )
                    #print(mappingList)

                    variable_check_list = []
                    variable_Parallel_value = [0] *  (variable_number + 1 + Aux)
                    clause_Parallel_value = [1] * (EndID + 1 - StartID)
                    
                    CNF_data_update = []
                    for j in range(EndID + 1):
                        CNF_data_update.append(CNF_data[j])
                    
                    tt = variable_Parallel_value_max 
                    for i in range(StartID, EndID + 1):
                        for j in range(len(CNF_data[i])):
                            if ((abs(CNF_data[i][j])-1) not in variable_check_list): #不在序列variable_check_list则增加，并记入 variable_Parallel_value
                                variable_check_list.append((abs(CNF_data[i][j])-1))
                                variable_Parallel_value[abs(CNF_data[i][j])-1] += 1
                                if (variable_Parallel_value[abs(CNF_data[i][j])-1] ) > clause_Parallel_value[i-StartID]: 
                                    clause_Parallel_value[i-StartID] = (variable_Parallel_value[abs(CNF_data[i][j])-1] % tt) 
                            else: #已在序列variable_check_list 
                                if (abs(CNF_data[i][j])-1) not in mapping_variable: #在序列variable_check_list但不在 qmemo 中，则增加 variable_Parallel_value
                                    variable_Parallel_value[abs(CNF_data[i][j])-1] += 1
                                    if (variable_Parallel_value[abs(CNF_data[i][j])-1] ) > clause_Parallel_value[i-StartID]: 
                                        clause_Parallel_value[i-StartID] = (variable_Parallel_value[abs(CNF_data[i][j])-1] % tt) 
                                else: #在序列variable_check_list 且 在 qmemo 中则增加，并记入到 合适的 variable_Parallel_value
                                    variable_Parallel_value[abs(CNF_data[i][j])-1] += 1
                                    if  variable_Parallel_value[abs(CNF_data[i][j])-1] < tt + 1:
                                        if (variable_Parallel_value[abs(CNF_data[i][j])-1]) > clause_Parallel_value[i-StartID]: 
                                            clause_Parallel_value[i-StartID] = (variable_Parallel_value[abs(CNF_data[i][j])-1] % tt)  
                                    if variable_Parallel_value[abs(CNF_data[i][j])-1] > tt:
                                        kk = 0
                                        for k in range(len(mappingList)):
                                            if mappingList[k][0] == (abs(CNF_data[i][j])-1):
                                                kk = k
                                                break
                                        kstep = math.ceil(variable_Parallel_value[abs(CNF_data[i][j])-1] / tt - 1)
                                        if variable_Parallel_value[abs(CNF_data[i][j])-1] % tt == 0:
                                            clause_Parallel_value[i-StartID] = tt
                                        else:
                                            if (variable_Parallel_value[abs(CNF_data[i][j])-1] % tt) > clause_Parallel_value[i-StartID]: 
                                                clause_Parallel_value[i-StartID] = (variable_Parallel_value[abs(CNF_data[i][j])-1] % tt)       
                                        #print(kk , kstep, CNF_data[i][j], variable_Parallel_value[abs(CNF_data[i][j])-1], tt,  len(mappingList) )
                                        #if CNF_data[i][j] > 0:
                                        #    CNF_data_update[i][j] = CNF_data[i][j] + mappingList[kk + kstep][1] - mappingList[kk + kstep][0]
                                        #else:
                                        #    if CNF_data[i][j] < 0:
                                        #        CNF_data_update[i][j] = CNF_data[i][j] - mappingList[kk + kstep][1] + mappingList[kk + kstep][0]
                        
                        Parallel_depth_list[(clause_Parallel_value[i-StartID]-1)].append(i)

                    cl_position = 0
                    #Auxqubit = [ ] * tt
                    for k in range(tt): #更新底层 并列的clause 安排
                        a1 = range( variable_number + 1 + Aux )
                        Auxqubit = set( list(a1) )
                        Auxqubit.remove(target)
                        for jjk in range(variable_number + 1 + CleanQubitNumber , variable_number + 1 + len(Parallel_depth_list[k])  +  CleanQubitNumber ):
                            Auxqubit.remove(jjk)
                        #各层记录分配辅助位的位置。
                        for kk in range(len(Parallel_depth_list[k])):
                            clasueID = Parallel_depth_list[k][kk]
                            controls = CNF_data_update[clasueID]
                            for i in range(len(controls)):
                                if controls[i] < 0  and  ((-controls[i]-1) in Auxqubit):
                                    Auxqubit.remove(-controls[i]-1)
                                if controls[i] > 0 and  ((controls[i]-1) in Auxqubit):
                                    Auxqubit.remove(controls[i]-1)
                        Auxqubit1 = list(Auxqubit)    

                        #分段做切片
                        for kk in range(len(Parallel_depth_list[k])):
                            clasueID = Parallel_depth_list[k][kk]
                            controls = CNF_data_update[clasueID]
                            controls_abs0=[]
                            controls_X0=[]
                            for i in range(len(controls)):
                                if controls[i] < 0:
                                    controls_abs0.append(-controls[i]-1)
                                if controls[i] > 0:
                                    controls_abs0.append(controls[i]-1)
                                    controls_X0.append(controls[i]-1)
                            for i in range(len(controls_X0)):
                                X | self._cgate(controls_X0[i])

                            dd =[]
                            #print( kk , clause_length , len(Auxqubit1), Auxqubit1)
                            for kkk in range(kk * clause_length, (kk + 1) * clause_length):
                                dd.append(Auxqubit1[kkk])
               
                            if controls_abs0 != []:
                                X | self._cgate(cl_position + CleanQubitNumber + 1 + variable_number)                                
                                MCTLinearHalfDirtyAux().execute(len(controls_abs0), len(controls_abs0) + clause_length +1 ) | self._cgate(controls_abs0 + dd + [ cl_position + CleanQubitNumber + 1 + variable_number ])
                                cl_position += 1
                            for i in range(len(controls_X0)):
                                X | self._cgate(controls_X0[i])
                            

    
                    b =  cl_position 
                    if EndID - StartID == 1:
                        CCX | self._cgate([variable_number + 1 + CleanQubitNumber, variable_number + 1 + 1 + CleanQubitNumber, target])
                    else:
                        for j in range(b-1):
                            #print(EndID - StartID,CleanQubitNumber,p,cl_position,variable_number + 1 + 2 * j + CleanQubitNumber , variable_number + 2 + 2 * j + CleanQubitNumber, variable_number + 1 + cl_position  + CleanQubitNumber)
                            CCX | self._cgate([variable_number + 1 + 2 * j + CleanQubitNumber , variable_number + 2 + 2 * j + CleanQubitNumber, variable_number + 1 + cl_position  + CleanQubitNumber])
                            cl_position += 1
                        CCX | self._cgate([variable_number - 1 + cl_position + CleanQubitNumber, variable_number + cl_position + CleanQubitNumber, target])
                        for j in range(b-2, -1 ,-1):
                            #print(EndID - StartID,CleanQubitNumber,p,cl_position,variable_number + 1 + 2 * j + CleanQubitNumber , variable_number + 2 + 2 * j + CleanQubitNumber, variable_number + 1 + cl_position  + CleanQubitNumber)
                            CCX | self._cgate([variable_number + 1 + 2 * j + CleanQubitNumber , variable_number + 2 + 2 * j + CleanQubitNumber, variable_number + cl_position  + CleanQubitNumber])
                            cl_position -= 1

                    #还原

                    cl_position = 0
                    #Auxqubit = [ ] * tt
                    for k in range(tt): #更新底层 并列的clause 安排
                        a1 = range( variable_number + 2 + Aux )
                        Auxqubit = set( list(a1) )
                        Auxqubit.remove(target)
                        for kk in range(variable_number + 1 + CleanQubitNumber , variable_number + 1 + 2 * CleanQubitNumber ):
                            Auxqubit.remove(kk)
                        #各层记录分配辅助位的位置。
                        for kk in range(len(Parallel_depth_list[k])):
                            clasueID = Parallel_depth_list[k][kk]
                            controls = CNF_data_update[clasueID]
                            for i in range(len(controls)):
                                if controls[i] < 0  and  ((-controls[i]-1) in Auxqubit):
                                    Auxqubit.remove(-controls[i]-1)
                                if controls[i] > 0 and  ((controls[i]-1) in Auxqubit):
                                    Auxqubit.remove(controls[i]-1)
                        Auxqubit1 = list(Auxqubit)    

                        #分段做切片
                        for kk in range(len(Parallel_depth_list[k])):
                            clasueID = Parallel_depth_list[k][kk]
                            controls = CNF_data_update[clasueID]
                            controls_abs0=[]
                            controls_X0=[]
                            for i in range(len(controls)):
                                if controls[i] < 0:
                                    controls_abs0.append(-controls[i]-1)
                                if controls[i] > 0:
                                    controls_abs0.append(controls[i]-1)
                                    controls_X0.append(controls[i]-1)
                            for i in range(len(controls_X0)):
                                X | self._cgate(controls_X0[i])

                            dd =[]
                            for kkk in range(kk * clause_length, (kk + 1) * clause_length):
                                dd.append(Auxqubit1[kkk])
               
                            if controls_abs0 != []:
                                X | self._cgate(cl_position + CleanQubitNumber + 1 + variable_number)                                
                                MCTLinearHalfDirtyAux().execute(len(controls_abs0), len(controls_abs0) + clause_length +1 ) | self._cgate(controls_abs0 + dd + [ cl_position + CleanQubitNumber + 1 + variable_number ])
                                cl_position += 1
                            for i in range(len(controls_X0)):
                                X | self._cgate(controls_X0[i])
                            
            else:
                #EndID-StartID 比较大，底层放不下，  block number >1
                block_len = math.ceil((EndID - StartID +1) /p)
                block_number = math.ceil((EndID - StartID + 1) / block_len )
                if block_number == 2:
                    if ((depth - current_depth) % 2) == 1: # 当前为奇数层  target 在variable_number + ancilla_qubits_num - p + 1 + j
                        
                        CCX | self._cgate([variable_number + 1  , variable_number  , target])    
                        self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, StartID, StartID + block_len -1 , variable_number + 1, current_depth-1, depth)

                        CCX | self._cgate([variable_number + 1  , variable_number  , target])
                        self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, StartID + block_len, EndID, variable_number , current_depth-1, depth)
                        
                        CCX | self._cgate([variable_number + 1  , variable_number  , target])
                        self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, StartID, StartID + block_len -1, variable_number + 1, current_depth-1, depth)

                        CCX | self._cgate([variable_number + 1  , variable_number  , target]) 
                        self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, StartID + block_len, EndID, variable_number , current_depth-1, depth)
                    else: # 当前偶数层 
                        
                        CCX | self._cgate([variable_number + CleanQubitNumber -1  , variable_number + CleanQubitNumber , target])   
                        self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, StartID, StartID + block_len-1, variable_number + CleanQubitNumber - 1, current_depth-1, depth)

                        CCX | self._cgate([variable_number + CleanQubitNumber -1  , variable_number + CleanQubitNumber , target])
                        self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, StartID + block_len, EndID, variable_number + CleanQubitNumber , current_depth-1, depth)
                        
                        CCX | self._cgate([variable_number + CleanQubitNumber -1  , variable_number + CleanQubitNumber , target])
                        self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, StartID, StartID + block_len-1, variable_number + CleanQubitNumber - 1, current_depth-1, depth)

                        CCX | self._cgate([variable_number + CleanQubitNumber -1  , variable_number + CleanQubitNumber , target]) 
                        self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, StartID + block_len, EndID, variable_number + CleanQubitNumber , current_depth-1, depth)

                else:   #block number >2
                    c=[]
                    for i in range( variable_number, variable_number + CleanQubitNumber  + 1 ):
                        if i != target :
                            c.append(i)
                    if ((depth - current_depth) % 2) == 1: 

                            #层数差 奇数 的存储位 为 variable_number +CleanQubitNumber- p+1+j  至 variable_number + CleanQubitNumber   要从差为偶数层 取数据
                            #层数差 偶数 的存储位 为 variable_number 至 variable_number + p -1      要从差为奇数层 取数据
                            # UpPhase 1 升阶段 第一位要单独处理，其target 即最终target。控制位一个在variable_number + Aux；另一个在 variable_number + CleanQubitNumber -  block_number +2 上， variable_number + CleanQubitNumber -  block_number +2 上放一个低一层的 clause。

                        CCX | self._cgate([c[block_number-1] , c[2*(block_number-1)-1] , target])
                        self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, 
                                    StartID, StartID + block_len -1 , c[block_number-1], current_depth-1, depth)
                        CCX | self._cgate([c[block_number-1] , c[2*(block_number-1)-1] , target])
                            
                            #控制位variable_number + CleanQubitNumber -  (block_number-1) + 2 -j 放 clause ， 另一个 控制位(将与此for内的前一个target 相同)并和target 依次上升
                        for j in range(1, block_number-2):
                            CCX | self._cgate([c[(block_number-1)  -j] , c[2*(block_number-1)-1  - j], c[2*(block_number-1) - j]])
                            self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber,
                                    StartID + j*block_len , StartID -1 + (j+1)*block_len, c[(block_number-1)-j], current_depth-1, depth)
                            CCX | self._cgate([c[(block_number-1)  -j] , c[2*(block_number-1)-1  - j], c[2*(block_number-1) - j]])
                                
                            # topPhase 
                        CCX | self._cgate([c[1]  , c[0], c[ block_number ]])    
                        self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber,  StartID + (block_number-2)*block_len, StartID + (block_number-1)*block_len -1, c[1], current_depth-1, depth)

                        CCX | self._cgate([c[1]  , c[0], c[ block_number ]])    
                        self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, StartID + (block_number-1)*block_len, EndID, c[0], current_depth-1, depth)

                        CCX | self._cgate([c[1]  , c[0], c[ block_number ]])    
                        self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, StartID + (block_number-2)*block_len, StartID + (block_number-1)*block_len -1, c[1], current_depth-1, depth)
            
                        CCX | self._cgate([c[1]  , c[0], c[ block_number ]])    
                        
                            #downPhase
                        for j in range(block_number-1 - 2 , 0, -1):
                            CCX | self._cgate([c[(block_number-1)  -j] , c[2*(block_number-1)-1  - j], c[2*(block_number-1) - j]])
                            self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, StartID + j*block_len , StartID -1 + (j+1)*block_len, c[(block_number-1)  -j], current_depth-1, depth)
                            CCX | self._cgate([c[(block_number-1)  -j] , c[2*(block_number-1)-1  - j], c[2*(block_number-1) - j]])   

                        CCX | self._cgate([c[block_number-1 ] , c[2*(block_number-1)-1] , target])
                        self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, StartID, StartID + block_len -1 , c[block_number-1 ], current_depth-1, depth)
                        CCX | self._cgate([c[ block_number-1] , c[2*(block_number-1)-1] , target])

                            #repeat....

                            # 还原各个位置
                            #控制位variable_number + CleanQubitNumber -  (block_number-1) + 2 -j 放 clause ， 另一个 控制位(将与此for内的前一个target 相同)并和target 依次上升
                        for j in range(1, block_number-1 - 1):
                            CCX | self._cgate([c[(block_number-1)  -j] , c[2*(block_number-1)-1  - j], c[2*(block_number-1) - j]])
                            self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, StartID + j*block_len , StartID -1 + (j+1)*block_len, c[(block_number-1)  -j], current_depth-1, depth)
                            CCX | self._cgate([c[(block_number-1)  -j] , c[2*(block_number-1)-1  - j], c[2*(block_number-1) - j]])
                                
                            # topPhase 
                        CCX | self._cgate([c[1]  , c[0], c[ block_number ]])    
                        self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, StartID + (block_number-2)*block_len, StartID + (block_number-1)*block_len -1, c[1], current_depth-1, depth)

                        CCX | self._cgate([c[1]  , c[0], c[ block_number ]])    
                        self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, StartID + (block_number-1)*block_len, EndID, c[0], current_depth-1, depth)

                        CCX | self._cgate([c[1]  , c[0], c[ block_number ]])    
                        self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, StartID + (block_number-2)*block_len, StartID + (block_number-1)*block_len -1, c[1], current_depth-1, depth)
            
                        CCX | self._cgate([c[1]  , c[0], c[ block_number ]])    
                        
                            #downPhase
                        for j in range(block_number-1 - 2 , 0, -1):
                            CCX | self._cgate([c[(block_number-1)  -j] , c[2*(block_number-1)-1  - j], c[2*(block_number-1) - j]])
                            self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, StartID + j*block_len , StartID -1 + (j+1)*block_len, c[(block_number-1)  -j], current_depth-1, depth)
                            CCX | self._cgate([c[(block_number-1)  -j] , c[2*(block_number-1)-1  - j], c[2*(block_number-1) - j]])   

                    else: #偶数层
                        CCX | self._cgate([c[CleanQubitNumber  -block_number ] , c[CleanQubitNumber - 2*(block_number-1)] , target])
                        self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, StartID, StartID -1 + block_len , c[CleanQubitNumber  -block_number ], current_depth-1, depth)
                        CCX | self._cgate([c[CleanQubitNumber  -block_number] , c[CleanQubitNumber - 2*(block_number-1)] , target])
                            
                            #控制位variable_number + CleanQubitNumber -  (block_number-1) + 2 -j 放 clause ， 另一个 控制位(将与此for内的前一个target 相同)并和target 依次上升
                        for j in range(1, block_number-2):
                            CCX | self._cgate([c[CleanQubitNumber-(block_number-1)+j-1] , c[CleanQubitNumber- 2*(block_number-1) + j], c[CleanQubitNumber- 2*(block_number-1) -1 + j]])
                            self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, StartID + j*block_len , StartID -1 +(1+ j)*block_len, c[CleanQubitNumber-(block_number-1)+j-1], current_depth-1, depth)
                            CCX | self._cgate([c[CleanQubitNumber-(block_number-1)+j-1] , c[CleanQubitNumber- 2*(block_number-1) + j], c[CleanQubitNumber- 2*(block_number-1) -1 + j]])
                                
                            # topPhase 
                        CCX | self._cgate([c[CleanQubitNumber-2]  , c[CleanQubitNumber-1], c[ CleanQubitNumber-2 - (block_number-1) ]])    
                        self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, StartID + (block_number-2)*block_len, StartID + (block_number-1)*block_len -1, c[CleanQubitNumber-2], current_depth-1, depth)

                        CCX | self._cgate([c[CleanQubitNumber-2]  , c[CleanQubitNumber-1], c[ CleanQubitNumber-2 - (block_number-1) ]])     
                        self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, StartID + (block_number-1)*block_len, EndID, c[CleanQubitNumber-1], current_depth-1, depth)

                        CCX | self._cgate([c[CleanQubitNumber-2]  , c[CleanQubitNumber-1], c[ CleanQubitNumber-2 - (block_number-1) ]])  
                        self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, StartID + (block_number-2)*block_len, StartID + (block_number-1)*block_len -1, c[CleanQubitNumber-2], current_depth-1, depth)
                        
                        CCX | self._cgate([c[CleanQubitNumber-2]  , c[CleanQubitNumber-1], c[ CleanQubitNumber-2 - (block_number-1) ]])    
                            
                            #控制位variable_number + CleanQubitNumber -  (block_number-1) + 2 -j 放 clause ， 另一个 控制位(将与此for内的前一个target 相同)并和target 依次上升
                        for j in range( block_number-3, 0 , -1):
                            CCX | self._cgate([c[CleanQubitNumber-(block_number-1)+j-1] , c[CleanQubitNumber- 2*(block_number-1) + j], c[CleanQubitNumber- 2*(block_number-1) -1 + j]])
                            self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, StartID + j*block_len , StartID -1 +(1+ j)*block_len, c[CleanQubitNumber-(block_number-1)+j-1], current_depth-1, depth)
                            CCX | self._cgate([c[CleanQubitNumber-(block_number-1)+j-1] , c[CleanQubitNumber- 2*(block_number-1) + j], c[CleanQubitNumber- 2*(block_number-1) -1 + j]])

                        CCX | self._cgate([c[CleanQubitNumber - block_number ] , c[CleanQubitNumber -  2*(block_number-1)] , target])
                        self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, StartID, StartID -1 + block_len , c[CleanQubitNumber - block_number ], current_depth-1, depth)
                        CCX | self._cgate([c[CleanQubitNumber - block_number] , c[CleanQubitNumber -  2*(block_number-1)] , target])

                            # 还原各个位置

                            #控制位variable_number + CleanQubitNumber -  (block_number-1) + 2 -j 放 clause ， 另一个 控制位(将与此for内的前一个target 相同)并和target 依次上升
                        for j in range(1, block_number-1 - 1):
                            CCX | self._cgate([c[CleanQubitNumber-(block_number-1)+j-1] , c[CleanQubitNumber- 2*(block_number-1) + j], c[CleanQubitNumber- 2*(block_number-1) -1 + j]])
                            self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, StartID + j*block_len , StartID -1 +(1+ j)*block_len, c[CleanQubitNumber-(block_number-1)+j-1], current_depth-1, depth)
                            CCX | self._cgate([c[CleanQubitNumber-(block_number-1)+j-1] , c[CleanQubitNumber- 2*(block_number-1) + j], c[CleanQubitNumber- 2*(block_number-1) -1 + j]])
                                
                            # topPhase 
                        CCX | self._cgate([c[CleanQubitNumber-2]  , c[CleanQubitNumber-1], c[ CleanQubitNumber-2 - (block_number-1) ]])    
                        self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, StartID + (block_number-2)*block_len, StartID + (block_number-1)*block_len -1, c[CleanQubitNumber-2], current_depth-1, depth)

                        CCX | self._cgate([c[CleanQubitNumber-2]  , c[CleanQubitNumber-1], c[ CleanQubitNumber-2 - (block_number-1) ]])     
                        self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, StartID + (block_number-1)*block_len, EndID, c[CleanQubitNumber-1], current_depth-1, depth)

                        CCX | self._cgate([c[CleanQubitNumber-2]  , c[CleanQubitNumber-1], c[ CleanQubitNumber-2 - (block_number-1) ]])  
                        self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, StartID + (block_number-2)*block_len, StartID + (block_number-1)*block_len -1, c[CleanQubitNumber-2], current_depth-1, depth)
                        
                        CCX | self._cgate([c[CleanQubitNumber-2]  , c[CleanQubitNumber-1], c[ CleanQubitNumber-2 - (block_number-1) ]])    
                            
                            #控制位variable_number + CleanQubitNumber -  (block_number-1) + 2 -j 放 clause ， 另一个 控制位(将与此for内的前一个target 相同)并和target 依次上升
                        for j in range( block_number-3, 0 , -1):
                            CCX | self._cgate([c[CleanQubitNumber-(block_number-1)+j-1] , c[CleanQubitNumber- 2*(block_number-1) + j], c[CleanQubitNumber- 2*(block_number-1) -1 + j]])
                            self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, StartID + j*block_len , StartID -1 +(1+ j)*block_len, c[CleanQubitNumber-(block_number-1)+j-1], current_depth-1, depth)
                            CCX | self._cgate([c[CleanQubitNumber-(block_number-1)+j-1] , c[CleanQubitNumber- 2*(block_number-1) + j], c[CleanQubitNumber- 2*(block_number-1) -1 + j]])
