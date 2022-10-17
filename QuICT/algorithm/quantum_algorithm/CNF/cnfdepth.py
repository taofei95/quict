# -*- coding:utf8 -*-
# @TIME    : 2022/7/
# @Author  : 
# @File    : 
from builtins import print
import math
import numpy as np
from QuICT.core import *     #Circuit
from QuICT.core.gate import *
from QuICT.qcda.synthesis.mct import MCTOneAux
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
        CleanQubitNumber = math.floor((ancilla_qubits_num) / (S+1))   #干净、脏比特数目   非底层能处理的块数
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
            
            current_Aux = target + 1
            for i in range(len(controls)):
                if controls[i] < 0:
                    controls_abs.append(-controls[i]-1)
                if controls[i] > 0:
                    controls_abs.append(controls[i]-1)
                    controls_X.append(controls[i]-1)
            for i in range(len(controls_X)):
                X | self._cgate(controls_X[i])
            X | self._cgate(target)
            if controls_abs != []:
                MCTOneAux().execute(len(controls_abs) + 2) | self._cgate(controls_abs + [ target , current_Aux]) 
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
                #print(controls_abs)
                #print(target)
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
                #print(controls_abs)
                #print(target)
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
                    #print(controls_abs)
                    #print(target)
                    if controls_abs != []:
                        MCTOneAux().execute(len(controls_abs) + 2) | self._cgate(controls_abs + [ c[j-1]  , c[j-1] + clause_number ])
                    # one_dirty_aux(self._cgate, controls_abs, target, current_Aux)
                    #X | self._cgate(target)
                    for i in range(len(controls_X)):
                        X | self._cgate(controls_X[i])
                
            else:
                p = math.floor( (CleanQubitNumber + 1) /2)
                block_len = math.ceil(clause_number / p)
                block_number = math.ceil(clause_number / block_len )
                
                controls = []
                
                for j in range(block_number):
                    self.clause(
                        CNF_data, variable_number, ancilla_qubits_num, clause_length, CleanQubitNumber,
                         j * block_len +1, np.minimum( (j+1) * block_len, clause_number),
                        variable_number + CleanQubitNumber - p + 1 + j, depth-1, depth
                    )
                    controls.append(variable_number + CleanQubitNumber - p + 1 + j)

                current_Aux = variable_number + 1 
                if controls != []:
                    MCTOneAux().execute(len(controls) + 2) | self._cgate(controls + [ target, current_Aux] ) 
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
        #print(CNF_data)
        f.close()

        return variable_number, clause_number, clause_length, CNF_data

    def clause(self, CNF_data: List, variable_number: int, Aux: int, clause_length :int, CleanQubitNumber  :int, StartID: int, EndID: int, target: int, current_depth: int, depth: int):

        p = math.floor((CleanQubitNumber + 1)/2) 
        if StartID == EndID: 
            #n= variable_number + Aux + 1
            controls = CNF_data[StartID]
            #if target > variable_number + p:
            #    current_Aux = target - 1
            #else:
            #    current_Aux = target + 1
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
            #print(controls_abs)
            #print(target)
            if controls_abs != []:
                MCTOneAux().execute(len(controls_abs) + 2) | self._cgate(controls_abs + [ target, target + CleanQubitNumber  ])
            # one_dirty_aux(self._cgate, controls_abs, target, current_Aux)
            #X | self._cgate(target)
            for i in range(len(controls_X)):
                X | self._cgate(controls_X[i])
        else:  #StartID 和 EndID 不同 
            #print(StartID, EndID)
            if ((EndID - StartID) * (clause_length) < (Aux - 2 * CleanQubitNumber + 1)) and ((EndID - StartID ) < p) : #if block_number == 1 而且 EndID - StartID > 1 
                #print(" <p ", EndID,StartID,p,target) 
                #print(c)
                #print(c,p, EndID,StartID, 2*(EndID - StartID)-1)
                
                c=[]
                for i in range(variable_number, variable_number + CleanQubitNumber +1):
                    if (i != target):
                        c.append(i)

                for j in range(StartID + 1, EndID + 1, 1):
                    controls = CNF_data[j]
                    for jj in range(len(controls)):
                        CX | self._cgate([abs(controls[jj])-1, (j - Start) * clause_length  + jj + variable_number + 2 * CleanQubitNumber +1])


                controls = CNF_data[StartID]
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
                X | self._cgate(target)
                #print(controls_abs)
                #print(target)
                if controls_abs0 != []:
                    MCTOneAux().execute(len(controls_abs0) + 2) | self._cgate(controls_abs0 + [ c[0], c[0] + CleanQubitNumber ])
                # one_dirty_aux(self._cgate, controls_abs, target, current_Aux)
                #X | self._cgate(target)
                for i in range(len(controls_X0)):
                    X | self._cgate(controls_X0[i])

                for j in range(StartID + 1, EndID + 1):
                    controls = CNF_data[j]
                    controls_abs=[]
                    controls_X=[]
                    for jj in range(len(controls)):
                        if controls[jj] < 0:
                            controls_abs.append((j - Start) * clause_length   + jj + variable_number + 2 * CleanQubitNumber +1)
                        if controls[jj] > 0:
                            controls_abs.append((j - Start) * clause_length + jj + variable_number + 2 * CleanQubitNumber +1)
                            controls_X.append((j - Start) * clause_length + jj + variable_number + 2 * CleanQubitNumber +1)
                    for i in range(len(controls_X)):
                        X | self._cgate(controls_X[i])
                    X | self._cgate(c[j - StartID])
                    #print(controls_abs)
                    #print(target)
                    if controls_abs != []:
                        MCTOneAux().execute(len(controls_abs) + 2) | self._cgate(controls_abs + [ c[j] , c[j] + CleanQubitNumber ])
                    # one_dirty_aux(self._cgate, controls_abs, target, current_Aux)
                    #X | self._cgate(target)
                    for i in range(len(controls_X)):
                        X | self._cgate(controls_X[i])
                a = 0
                b = EndID - Start + 1  
                if b == 2:
                    CCX | self._cgate([c[0], c[1], target])
                else:
                    for j in range(EndID - StartID):
                        CCX | self._cgate([c[a + 2*j], c[a + 2*j + 1], c[b]])
                        b = b + 1
                    CCX | self._cgate([c[b-2], c[b-1], target])


                #还原
                for j in range(StartID + 1, EndID + 1 ):
                    controls = CNF_data[j]
                    for jj in range(len(controls)):
                        CX | self._cgate([abs(controls[jj])-1, (j - Start) * clause_length  + jj + variable_number + 2 * CleanQubitNumber +1])

                controls = CNF_data[StartID]
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
                X | self._cgate(target)
                #print(controls_abs)
                #print(target)
                if controls_abs0 != []:
                    MCTOneAux().execute(len(controls_abs0) + 2) | self._cgate(controls_abs0 + [ c[0] , c[0] + CleanQubitNumber ])
                # one_dirty_aux(self._cgate, controls_abs, target, current_Aux)
                #X | self._cgate(target)
                for i in range(len(controls_X0)):
                    X | self._cgate(controls_X0[i])

                for j in range(StartID + 1, EndID + 1):
                    controls = CNF_data[j]
                    controls_abs=[]
                    controls_X=[]
                    for jj in range(len(controls)):
                        if controls[jj] < 0:
                            controls_abs.append((j - Start) * clause_length   + jj + variable_number + 2 * CleanQubitNumber +1)
                        if controls[jj] > 0:
                            controls_abs.append((j - Start) * clause_length + jj + variable_number + 2 * CleanQubitNumber +1)
                            controls_X.append((j - Start) * clause_length + jj + variable_number + 2 * CleanQubitNumber +1)
                    for i in range(len(controls_X)):
                        X | self._cgate(controls_X[i])
                    X | self._cgate(c[j - StartID])
                    #print(controls_abs)
                    #print(target)
                    if controls_abs != []:
                        MCTOneAux().execute(len(controls_abs) + 2) | self._cgate(controls_abs + [ c[j] , c[j]  + CleanQubitNumber ])
                    # one_dirty_aux(self._cgate, controls_abs, target, current_Aux)
                    #X | self._cgate(target)
                    for i in range(len(controls_X)):
                        X | self._cgate(controls_X[i])

            else: #EndID-StartID 比较大，底层放不下，  block number >1
                block_len = math.ceil((EndID - StartID +1) /p)
                block_number = math.ceil((EndID - StartID + 1) / block_len )
                if block_number == 2:
                    if ((depth - current_depth) % 2) == 1: # 当前为奇数层  target 在variable_number + ancilla_qubits_num - p + 1 + j
                        #print("2 block 1", target)
                        CCX | self._cgate([variable_number + 1  , variable_number  , target])    
                        self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, StartID, StartID + block_len -1 , variable_number + 1, current_depth-1, depth)

                        CCX | self._cgate([variable_number + 1  , variable_number  , target])
                        self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, StartID + block_len, EndID, variable_number , current_depth-1, depth)
                        
                        CCX | self._cgate([variable_number + 1  , variable_number  , target])
                        self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, StartID, StartID + block_len -1, variable_number + 1, current_depth-1, depth)

                        CCX | self._cgate([variable_number + 1  , variable_number  , target]) 
                        self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, StartID + block_len, EndID, variable_number , current_depth-1, depth)
                    else: # 当前偶数层 
                        #print("2 block 0", target)  
                        CCX | self._cgate([variable_number + CleanQubitNumber -1  , variable_number + CleanQubitNumber , target])   
                        self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, StartID, StartID + block_len-1, variable_number + CleanQubitNumber - 1, current_depth-1, depth)

                        CCX | self._cgate([variable_number + CleanQubitNumber -1  , variable_number + CleanQubitNumber , target])
                        self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, StartID + block_len, EndID, variable_number + CleanQubitNumber , current_depth-1, depth)
                        
                        CCX | self._cgate([variable_number + CleanQubitNumber -1  , variable_number + CleanQubitNumber , target])
                        self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, StartID, StartID + block_len-1, variable_number + CleanQubitNumber - 1, current_depth-1, depth)

                        CCX | self._cgate([variable_number + CleanQubitNumber -1  , variable_number + CleanQubitNumber , target]) 
                        self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, StartID + block_len, EndID, variable_number + CleanQubitNumber , current_depth-1, depth)
                else:
                        #block number >2
                    c=[]
                    for i in range( variable_number, variable_number + CleanQubitNumber  + 1 ):
                        if i != target :
                            c.append(i)
                    if ((depth - current_depth) % 2) == 1: 
                            #print(   "2n"   )   
                            #层数差 奇数 的存储位 为 variable_number +CleanQubitNumber- p+1+j  至 variable_number + CleanQubitNumber   要从差为偶数层 取数据
                            #层数差 偶数 的存储位 为 variable_number 至 variable_number + p -1      要从差为奇数层 取数据
                            # UpPhase 1 升阶段 第一位要单独处理，其target 即最终target。控制位一个在variable_number + Aux；另一个在 variable_number + CleanQubitNumber -  block_number +2 上， variable_number + CleanQubitNumber -  block_number +2 上放一个低一层的 clause。
                        #print(EndID ,StartID,p,block_len)
                        #print(Aux,len(c),block_number,"aabb" )
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
                            #print(Aux, block_number,j,EndID,StartID, CleanQubitNumber-(block_number-1)+j-1,  CleanQubitNumber- 2*(block_number-1) + j ,CleanQubitNumber- 2*(block_number-1) -1 + j,len(c))
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


