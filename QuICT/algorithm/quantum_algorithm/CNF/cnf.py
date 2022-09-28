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
from QuICT.qcda.synthesis.mct import one_dirty_aux
from QuICT.qcda.synthesis.mct.mct_linear_simulation import half_dirty_aux
from QuICT.qcda.optimization.commutative_optimization import *

#from .._synthesis import Synthesis
#import random import logging from math import pi, gcd
#from fractions import Fraction
#pyfrom sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from QuICT.core import Circuit
#from QuICT.core.gate import *cc
# from QuICT.simulation.cpu_simulator import CircuitSimulator
#from QuICT.qcda.optimization._optimization import Optimization

class CNFSATOracle:
     
    def __init__(self, simu = None):
        self.simulator = simu

    def circuit(self) -> CompositeGate:
        return self._cgate
    
    def run(
        self,
        cnf_file: str,
        ancilla_qubits_num: int = 3
    ) -> int:
        """ Run CNF algorithm

        Args:
            cnf_file (str): The file path
            ancilla_qubits_num (int): >= 3
        """
        # check if Aux > 2
        assert ancilla_qubits_num > 2 , "Need at least 3 auxiliary qubit."

        # Step 1: Read CNF File
        variable_number, clause_number, CNF_data = self.read_CNF(cnf_file)

        # Step 2: Construct Circuit
        self._cgate = CompositeGate()
        p = math.floor((ancilla_qubits_num+1) / 2) 
        depth=math.ceil(math.log( clause_number , p ))
        target = variable_number
        if clause_number==1:
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
            one_dirty_aux(self._cgate, controls_abs, target, current_Aux) #QuICT.qcda.synthesis.mct.
            #X | self._cgate(target)
            for i in range(len(controls_X)):
                X | self._cgate(controls_X[i])
        else:
            if clause_number < ancilla_qubits_num+1 :
                controls_abs=[]
                for j in range(clause_number):
                    controls_abs.append(variable_number +  j + 1)
                    self.clause(
                        CNF_data, variable_number,
                        ancilla_qubits_num, j+1, j+1,
                        variable_number +  j+1, depth-1, depth
                    )
                one_dirty_aux(self._cgate, controls_abs, target, target-1)
                for j in range(clause_number):
                    controls_abs.append(variable_number +  j+1)
                    self.clause(
                        CNF_data, variable_number,
                        ancilla_qubits_num, j+1, j+1,
                        variable_number +  j+1, depth-1, depth
                    )
            else:
                block_len = p ** (math.ceil(math.log(clause_number,p))-1)
                block_number = math.ceil(clause_number / block_len )
                controls = []
                #if (math.floor((depth - current_depth) % 2)) == 1:    #math.floor(math.log(EndID-StartID,p))
                for j in range(block_number):
                    self.clause(
                        CNF_data, variable_number,
                        ancilla_qubits_num, j * block_len +1, np.minimum( (j+1) * block_len, clause_number),
                        variable_number + ancilla_qubits_num - p + 1 + j, depth-1, depth
                    )
                    controls.append(variable_number + ancilla_qubits_num - p + 1 + j)

                current_Aux = variable_number + 1 
                one_dirty_aux(self._cgate, controls, target, current_Aux)
                
                for j in range(block_number):
                    self.clause(
                        CNF_data, variable_number,
                        ancilla_qubits_num,  j * block_len +1, np.minimum( (j+1) * block_len, clause_number),
                        variable_number + ancilla_qubits_num - p + 1 + j, depth-1, depth
                    )

    def read_CNF(self, cnf_file):
        # file analysis
        #输入一个文件，输出CNF格式的list
        variable_number = 0
        clause_number = 0
        CNF_data = []
        f = open(cnf_file, 'r') 
        for line in f.readlines():
            new = line.strip().split()
            int_new=[]
            if new[0] == 'p':
                variable_number = int(new[2])
                clause_number = int(new[3])
            else:
                for i in range(len(new)-1): #注意这里是否减1 要检查一下
                    int_new.append(int(new[i]))
            CNF_data.append(int_new)  #给各个Clause 编号0,1 ...m-1#

        f.close()

        return variable_number, clause_number, CNF_data

    def clause(self, CNF_data: List, variable_number: int, Aux: int, StartID: int, EndID: int, target: int, current_depth: int, depth: int):

        p = math.floor((Aux+1)/2) 
        if StartID == EndID: 
            #n= variable_number + Aux + 1
            controls = CNF_data[StartID]
            if target > variable_number + p:
                current_Aux = target - 1
            else:
                current_Aux = target + 1
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
            one_dirty_aux(self._cgate, controls_abs, target, current_Aux)
            #X | self._cgate(target)
            for i in range(len(controls_X)):
                X | self._cgate(controls_X[i])
        else:  #StartID 和 EndID 不同 
            if  (EndID - StartID) == 1 : #StartID +1 等于 EndID            
                if ((depth - current_depth) % 2) == 1: # 当前为奇数数层 
                    #print(" ==1 ", EndID,StartID,p)
                    
                    CCX | self._cgate([variable_number + 1  , variable_number  , target])    
                    self.clause(CNF_data, variable_number, Aux, StartID,StartID, variable_number + 1, current_depth-1, depth)

                    CCX | self._cgate([variable_number + 1  , variable_number  , target])
                    self.clause(CNF_data, variable_number, Aux, EndID, EndID, variable_number , current_depth-1, depth)
                    
                    CCX | self._cgate([variable_number + 1  , variable_number  , target])
                    self.clause(CNF_data, variable_number, Aux, StartID,StartID, variable_number + 1, current_depth-1, depth)

                    CCX | self._cgate([variable_number + 1  , variable_number  , target])
                    self.clause(CNF_data, variable_number, Aux, EndID, EndID, variable_number , current_depth-1, depth)
                
                else: # 当前偶数层   target 在variable_number + ancilla_qubits_num - p + 1 + j
                    #print(" ==0 ", EndID,StartID,p)
                    CCX | self._cgate([variable_number + Aux -1  , variable_number + Aux , target])   
                    self.clause(CNF_data, variable_number, Aux, StartID,StartID, variable_number + Aux -1, current_depth-1, depth)

                    CCX | self._cgate([variable_number + Aux -1  , variable_number + Aux , target])
                    self.clause(CNF_data, variable_number, Aux, EndID, EndID, variable_number + Aux, current_depth-1, depth)
                    
                    CCX | self._cgate([variable_number + Aux -1  , variable_number + Aux , target])
                    self.clause(CNF_data, variable_number, Aux, StartID,StartID, variable_number + Aux -1, current_depth-1, depth)

                    CCX | self._cgate([variable_number + Aux -1  , variable_number + Aux , target]) 
                    self.clause(CNF_data, variable_number, Aux, EndID, EndID, variable_number + Aux, current_depth-1, depth)                
                
            else: #EndID - StartID > 1 
                if (EndID - StartID) < p : #if block_number == 1 而且 EndID - StartID > 1 
                    #print(" <p ", EndID,StartID,p,target) 
                    #print(c)
                    #print(c,p, EndID,StartID, 2*(EndID - StartID)-1)
                    c=[]
                    for i in range( variable_number, variable_number + Aux +1):
                        if (i != target):
                            c.append(i)
                   
                    if ((depth- current_depth) % 2) == 1 : 
                        # UpPhase 1 升阶段 第一位要单独处理，其target 即最终target。控制位一个在variable_number + Aux；另一个在 variable_number + Aux - block_number +2 上， variable_number + Aux - block_number +2 上放一个低一层的 clause。
                        #print(p, 2*(EndID - StartID)-1)
                        CCX | self._cgate([c[EndID - StartID] , c[2*(EndID - StartID)-1] , target])
                        self.clause(CNF_data, variable_number, Aux, StartID, StartID , c[EndID - StartID ], current_depth-1, depth)
                        CCX | self._cgate([c[EndID - StartID] , c[2*(EndID - StartID)-1] , target])
                            
                            #控制位variable_number + Aux - (EndID-StartID) + 2 -j 放 clause ， 另一个 控制位(将与此for内的前一个target 相同)并和target 依次上升
                        for j in range(1, EndID - StartID - 1):
                            CCX | self._cgate([c[(EndID-StartID)  -j] , c[2*(EndID - StartID)-1  - j], c[2*(EndID - StartID) - j]])
                            self.clause(CNF_data, variable_number, Aux, StartID + j , StartID + j, c[(EndID-StartID)  -j], current_depth-1, depth)
                            CCX | self._cgate([c[(EndID-StartID)  -j] , c[2*(EndID - StartID)-1  - j], c[2*(EndID - StartID) - j]])
                                
                            # topPhase 
                        CCX | self._cgate([c[1]  , c[0], c[ EndID - StartID +1]])    
                        self.clause(CNF_data, variable_number, Aux, (EndID-1), EndID-1, c[1], current_depth-1, depth)

                        CCX | self._cgate([c[1]  , c[0], c[ EndID - StartID +1]])    
                        self.clause(CNF_data, variable_number, Aux, EndID, EndID, c[0], current_depth-1, depth)

                        CCX | self._cgate([c[1]  , c[0], c[ EndID - StartID +1]])    
                        self.clause(CNF_data, variable_number, Aux, (EndID-1), EndID-1, c[1], current_depth-1, depth)
            
                        CCX | self._cgate([c[1]  , c[0], c[ EndID - StartID +1]])    
                        
                            #downPhase
                        for j in range(EndID - StartID - 2 , 0, -1):
                            CCX | self._cgate([c[(EndID-StartID)  -j] , c[2*(EndID - StartID)-1  - j], c[2*(EndID - StartID) - j]])
                            self.clause(CNF_data, variable_number, Aux, StartID + j , StartID + j, c[(EndID-StartID)  -j], current_depth-1, depth)
                            CCX | self._cgate([c[(EndID-StartID)  -j] , c[2*(EndID - StartID)-1  - j], c[2*(EndID - StartID) - j]])   

                        CCX | self._cgate([c[EndID - StartID ] , c[2*(EndID - StartID)-1] , target])
                        self.clause(CNF_data, variable_number, Aux, StartID, StartID , c[EndID - StartID ], current_depth-1, depth)
                        CCX | self._cgate([c[ EndID - StartID] , c[2*(EndID - StartID)-1] , target])

                            #repeat....

                            # 还原各个位置
                             #控制位variable_number + Aux - (EndID-StartID) + 2 -j 放 clause ， 另一个 控制位(将与此for内的前一个target 相同)并和target 依次上升
                        for j in range(1, EndID - StartID - 1):
                            CCX | self._cgate([c[(EndID-StartID)  -j] , c[2*(EndID - StartID)-1  - j], c[2*(EndID - StartID) - j]])
                            self.clause(CNF_data, variable_number, Aux, StartID + j , StartID + j, c[(EndID-StartID)  -j], current_depth-1, depth)
                            CCX | self._cgate([c[(EndID-StartID)  -j] , c[2*(EndID - StartID)-1  - j], c[2*(EndID - StartID) - j]])
                                
                            # topPhase 
                        CCX | self._cgate([c[1]  , c[0], c[ EndID - StartID +1]])    
                        self.clause(CNF_data, variable_number, Aux, (EndID-1), EndID-1, c[1], current_depth-1, depth)

                        CCX | self._cgate([c[1]  , c[0], c[ EndID - StartID +1]])    
                        self.clause(CNF_data, variable_number, Aux, EndID, EndID, c[0], current_depth-1, depth)

                        CCX | self._cgate([c[1]  , c[0], c[ EndID - StartID +1]])    
                        self.clause(CNF_data, variable_number, Aux, (EndID-1), EndID-1, c[1], current_depth-1, depth)
            
                        CCX | self._cgate([c[1]  , c[0], c[ EndID - StartID +1]])    
                        
                            #downPhase
                        for j in range(EndID - StartID - 2 , 0, -1):
                            CCX | self._cgate([c[(EndID-StartID)  -j] , c[2*(EndID - StartID)-1  - j], c[2*(EndID - StartID) - j]])
                            self.clause(CNF_data, variable_number, Aux, StartID + j , StartID + j, c[(EndID-StartID)  -j], current_depth-1, depth)
                            CCX | self._cgate([c[(EndID-StartID)  -j] , c[2*(EndID - StartID)-1  - j], c[2*(EndID - StartID) - j]]) 

                    else: #偶数层
                        CCX | self._cgate([c[Aux -1 - EndID + StartID ] , c[Aux - 2*(EndID - StartID)] , target])
                        self.clause(CNF_data, variable_number, Aux, StartID, StartID , c[Aux -1 - EndID + StartID ], current_depth-1, depth)
                        CCX | self._cgate([c[Aux -1 - EndID + StartID] , c[Aux - 2*(EndID - StartID)] , target])
                            
                            #控制位variable_number + Aux - (EndID-StartID) + 2 -j 放 clause ， 另一个 控制位(将与此for内的前一个target 相同)并和target 依次上升
                        for j in range(1, EndID - StartID - 1):
                            CCX | self._cgate([c[Aux-(EndID-StartID)+j-1] , c[Aux- 2*(EndID - StartID) + j], c[Aux- 2*(EndID - StartID) -1 + j]])
                            self.clause(CNF_data, variable_number, Aux, StartID + j , StartID + j, c[Aux-(EndID-StartID)+j-1], current_depth-1, depth)
                            CCX | self._cgate([c[Aux-(EndID-StartID)+j-1] , c[Aux- 2*(EndID - StartID) + j], c[Aux- 2*(EndID - StartID) -1 + j]])
                                
                            # topPhase 
                        CCX | self._cgate([c[Aux-2]  , c[Aux-1], c[ Aux-2 - EndID + StartID ]])    
                        self.clause(CNF_data, variable_number, Aux, (EndID-1), EndID-1, c[Aux-2], current_depth-1, depth)

                        CCX | self._cgate([c[Aux-2]  , c[Aux-1], c[ Aux-2 - EndID + StartID ]])     
                        self.clause(CNF_data, variable_number, Aux, EndID, EndID, c[Aux-1], current_depth-1, depth)

                        CCX | self._cgate([c[Aux-2]  , c[Aux-1], c[ Aux-2 - EndID + StartID ]])  
                        self.clause(CNF_data, variable_number, Aux, (EndID-1), EndID-1, c[Aux-2], current_depth-1, depth)
                        
                        CCX | self._cgate([c[Aux-2]  , c[Aux-1], c[ Aux-2 - EndID + StartID ]])    
                            
                            #控制位variable_number + Aux - (EndID-StartID) + 2 -j 放 clause ， 另一个 控制位(将与此for内的前一个target 相同)并和target 依次上升
                        for j in range( EndID - StartID - 2, 0 , -1):
                            CCX | self._cgate([c[Aux-(EndID-StartID)+j-1] , c[Aux- 2*(EndID - StartID) + j], c[Aux- 2*(EndID - StartID) -1 + j]])
                            self.clause(CNF_data, variable_number, Aux, StartID + j , StartID + j, c[Aux-(EndID-StartID)+j-1], current_depth-1, depth)
                            CCX | self._cgate([c[Aux-(EndID-StartID)+j-1] , c[Aux- 2*(EndID - StartID) + j], c[Aux- 2*(EndID - StartID) -1 + j]])

                        CCX | self._cgate([c[Aux -1 - EndID + StartID ] , c[Aux - 2*(EndID - StartID)] , target])
                        self.clause(CNF_data, variable_number, Aux, StartID, StartID , c[Aux -1 - EndID + StartID ], current_depth-1, depth)
                        CCX | self._cgate([c[Aux -1 - EndID + StartID] , c[Aux - 2*(EndID - StartID)] , target])

                            # 还原各个位置

                            #控制位variable_number + Aux - (EndID-StartID) + 2 -j 放 clause ， 另一个 控制位(将与此for内的前一个target 相同)并和target 依次上升
                        for j in range(1, EndID - StartID - 1):
                            CCX | self._cgate([c[Aux-(EndID-StartID)+j-1] , c[Aux- 2*(EndID - StartID) + j], c[Aux- 2*(EndID - StartID) -1 + j]])
                            self.clause(CNF_data, variable_number, Aux, StartID + j , StartID + j, c[Aux-(EndID-StartID)+j-1], current_depth-1, depth)
                            CCX | self._cgate([c[Aux-(EndID-StartID)+j-1] , c[Aux- 2*(EndID - StartID) + j], c[Aux- 2*(EndID - StartID) -1 + j]])
                                
                            # topPhase 
                        CCX | self._cgate([c[Aux-2]  , c[Aux-1], c[ Aux-2 - EndID + StartID ]])    
                        self.clause(CNF_data, variable_number, Aux, (EndID-1), EndID-1, c[Aux-2], current_depth-1, depth)

                        CCX | self._cgate([c[Aux-2]  , c[Aux-1], c[ Aux-2 - EndID + StartID ]])     
                        self.clause(CNF_data, variable_number, Aux, EndID, EndID, c[Aux-1], current_depth-1, depth)

                        CCX | self._cgate([c[Aux-2]  , c[Aux-1], c[ Aux-2 - EndID + StartID ]])  
                        self.clause(CNF_data, variable_number, Aux, (EndID-1), EndID-1, c[Aux-2], current_depth-1, depth)
                        
                        CCX | self._cgate([c[Aux-2]  , c[Aux-1], c[ Aux-2 - EndID + StartID ]])    
                            
                            #控制位variable_number + Aux - (EndID-StartID) + 2 -j 放 clause ， 另一个 控制位(将与此for内的前一个target 相同)并和target 依次上升
                        for j in range( EndID - StartID - 2, 0 , -1):
                            CCX | self._cgate([c[Aux-(EndID-StartID)+j-1] , c[Aux- 2*(EndID - StartID) + j], c[Aux- 2*(EndID - StartID) -1 + j]])
                            self.clause(CNF_data, variable_number, Aux, StartID + j , StartID + j, c[Aux-(EndID-StartID)+j-1], current_depth-1, depth)
                            CCX | self._cgate([c[Aux-(EndID-StartID)+j-1] , c[Aux- 2*(EndID - StartID) + j], c[Aux- 2*(EndID - StartID) -1 + j]])

                else: #EndID-StartID > p-1  block number >1
                    block_len = math.ceil((EndID - StartID +1) /p)
                    block_number = math.ceil((EndID - StartID + 1) / block_len )
                    if block_number == 2:
                        if ((depth - current_depth) % 2) == 1: # 当前为奇数层  target 在variable_number + ancilla_qubits_num - p + 1 + j
                            #print("2 block 1", target)
                            CCX | self._cgate([variable_number + 1  , variable_number  , target])    
                            self.clause(CNF_data, variable_number, Aux, StartID, StartID + block_len -1 , variable_number + 1, current_depth-1, depth)

                            CCX | self._cgate([variable_number + 1  , variable_number  , target])
                            self.clause(CNF_data, variable_number, Aux, StartID + block_len, EndID, variable_number , current_depth-1, depth)
                            
                            CCX | self._cgate([variable_number + 1  , variable_number  , target])
                            self.clause(CNF_data, variable_number, Aux, StartID, StartID + block_len -1, variable_number + 1, current_depth-1, depth)

                            CCX | self._cgate([variable_number + 1  , variable_number  , target]) 
                            self.clause(CNF_data, variable_number, Aux, StartID + block_len, EndID, variable_number , current_depth-1, depth)
                        else: # 当前偶数层 
                            #print("2 block 0", target)  
                            CCX | self._cgate([variable_number + Aux -1  , variable_number + Aux , target])   
                            self.clause(CNF_data, variable_number, Aux, StartID, StartID + block_len-1, variable_number + Aux -1, current_depth-1, depth)

                            CCX | self._cgate([variable_number + Aux -1  , variable_number + Aux , target])
                            self.clause(CNF_data, variable_number, Aux, StartID + block_len, EndID, variable_number + Aux, current_depth-1, depth)
                            
                            CCX | self._cgate([variable_number + Aux -1  , variable_number + Aux , target])
                            self.clause(CNF_data, variable_number, Aux, StartID, StartID + block_len-1, variable_number + Aux -1, current_depth-1, depth)

                            CCX | self._cgate([variable_number + Aux -1  , variable_number + Aux , target]) 
                            self.clause(CNF_data, variable_number, Aux, StartID + block_len, EndID, variable_number + Aux, current_depth-1, depth)
                    else:
                         #block number >2
                        c=[]
                        for i in range( variable_number, variable_number + Aux +1):
                            if i != target:
                                c.append(i)
                        if ((depth - current_depth) % 2) == 1: 
                                #print(   "2n"   )   
                                #层数差 奇数 的存储位 为 variable_number +Aux- p+1+j  至 variable_number + Aux  要从差为偶数层 取数据
                                #层数差 偶数 的存储位 为 variable_number 至 variable_number + p -1      要从差为奇数层 取数据
                               # UpPhase 1 升阶段 第一位要单独处理，其target 即最终target。控制位一个在variable_number + Aux；另一个在 variable_number + Aux - block_number +2 上， variable_number + Aux - block_number +2 上放一个低一层的 clause。
                            print(EndID ,StartID,p,block_len)
                            print(Aux,len(c),block_number,"aabb" )
                            CCX | self._cgate([c[block_number-1] , c[2*(block_number-1)-1] , target])
                            self.clause(CNF_data, variable_number, Aux, StartID, StartID + block_len -1 , c[block_number-1], current_depth-1, depth)
                            CCX | self._cgate([c[block_number-1] , c[2*(block_number-1)-1] , target])
                                
                                #控制位variable_number + Aux - (block_number-1) + 2 -j 放 clause ， 另一个 控制位(将与此for内的前一个target 相同)并和target 依次上升
                            for j in range(1, block_number-2):
                                CCX | self._cgate([c[(block_number-1)  -j] , c[2*(block_number-1)-1  - j], c[2*(block_number-1) - j]])
                                self.clause(CNF_data, variable_number, Aux, StartID + j*block_len , StartID -1 + (j+1)*block_len, c[(block_number-1)-j], current_depth-1, depth)
                                CCX | self._cgate([c[(block_number-1)  -j] , c[2*(block_number-1)-1  - j], c[2*(block_number-1) - j]])
                                    
                                # topPhase 
                            CCX | self._cgate([c[1]  , c[0], c[ block_number ]])    
                            self.clause(CNF_data, variable_number, Aux, StartID + (block_number-2)*block_len, StartID + (block_number-1)*block_len -1, c[1], current_depth-1, depth)

                            CCX | self._cgate([c[1]  , c[0], c[ block_number ]])    
                            self.clause(CNF_data, variable_number, Aux, StartID + (block_number-1)*block_len, EndID, c[0], current_depth-1, depth)

                            CCX | self._cgate([c[1]  , c[0], c[ block_number ]])    
                            self.clause(CNF_data, variable_number, Aux, StartID + (block_number-2)*block_len, StartID + (block_number-1)*block_len -1, c[1], current_depth-1, depth)
                
                            CCX | self._cgate([c[1]  , c[0], c[ block_number ]])    
                            
                                #downPhase
                            for j in range(block_number-1 - 2 , 0, -1):
                                CCX | self._cgate([c[(block_number-1)  -j] , c[2*(block_number-1)-1  - j], c[2*(block_number-1) - j]])
                                self.clause(CNF_data, variable_number, Aux, StartID + j*block_len , StartID -1 + (j+1)*block_len, c[(block_number-1)  -j], current_depth-1, depth)
                                CCX | self._cgate([c[(block_number-1)  -j] , c[2*(block_number-1)-1  - j], c[2*(block_number-1) - j]])   

                            CCX | self._cgate([c[block_number-1 ] , c[2*(block_number-1)-1] , target])
                            self.clause(CNF_data, variable_number, Aux, StartID, StartID + block_len -1 , c[block_number-1 ], current_depth-1, depth)
                            CCX | self._cgate([c[ block_number-1] , c[2*(block_number-1)-1] , target])

                                #repeat....

                                # 还原各个位置
                                #控制位variable_number + Aux - (block_number-1) + 2 -j 放 clause ， 另一个 控制位(将与此for内的前一个target 相同)并和target 依次上升
                            for j in range(1, block_number-1 - 1):
                                CCX | self._cgate([c[(block_number-1)  -j] , c[2*(block_number-1)-1  - j], c[2*(block_number-1) - j]])
                                self.clause(CNF_data, variable_number, Aux, StartID + j*block_len , StartID -1 + (j+1)*block_len, c[(block_number-1)  -j], current_depth-1, depth)
                                CCX | self._cgate([c[(block_number-1)  -j] , c[2*(block_number-1)-1  - j], c[2*(block_number-1) - j]])
                                    
                                # topPhase 
                            CCX | self._cgate([c[1]  , c[0], c[ block_number ]])    
                            self.clause(CNF_data, variable_number, Aux, StartID + (block_number-2)*block_len, StartID + (block_number-1)*block_len -1, c[1], current_depth-1, depth)

                            CCX | self._cgate([c[1]  , c[0], c[ block_number ]])    
                            self.clause(CNF_data, variable_number, Aux, StartID + (block_number-1)*block_len, EndID, c[0], current_depth-1, depth)

                            CCX | self._cgate([c[1]  , c[0], c[ block_number ]])    
                            self.clause(CNF_data, variable_number, Aux, StartID + (block_number-2)*block_len, StartID + (block_number-1)*block_len -1, c[1], current_depth-1, depth)
                
                            CCX | self._cgate([c[1]  , c[0], c[ block_number ]])    
                            
                                #downPhase
                            for j in range(block_number-1 - 2 , 0, -1):
                                CCX | self._cgate([c[(block_number-1)  -j] , c[2*(block_number-1)-1  - j], c[2*(block_number-1) - j]])
                                self.clause(CNF_data, variable_number, Aux, StartID + j*block_len , StartID -1 + (j+1)*block_len, c[(block_number-1)  -j], current_depth-1, depth)
                                CCX | self._cgate([c[(block_number-1)  -j] , c[2*(block_number-1)-1  - j], c[2*(block_number-1) - j]])   

                        else: #偶数层
                            CCX | self._cgate([c[Aux  -block_number ] , c[Aux - 2*(block_number-1)] , target])
                            self.clause(CNF_data, variable_number, Aux, StartID, StartID -1 + block_len , c[Aux  -block_number ], current_depth-1, depth)
                            CCX | self._cgate([c[Aux  -block_number] , c[Aux - 2*(block_number-1)] , target])
                                
                                #控制位variable_number + Aux - (block_number-1) + 2 -j 放 clause ， 另一个 控制位(将与此for内的前一个target 相同)并和target 依次上升
                            for j in range(1, block_number-2):
                                CCX | self._cgate([c[Aux-(block_number-1)+j-1] , c[Aux- 2*(block_number-1) + j], c[Aux- 2*(block_number-1) -1 + j]])
                                self.clause(CNF_data, variable_number, Aux, StartID + j*block_len , StartID -1 +(1+ j)*block_len, c[Aux-(block_number-1)+j-1], current_depth-1, depth)
                                CCX | self._cgate([c[Aux-(block_number-1)+j-1] , c[Aux- 2*(block_number-1) + j], c[Aux- 2*(block_number-1) -1 + j]])
                                    
                                # topPhase 
                            CCX | self._cgate([c[Aux-2]  , c[Aux-1], c[ Aux-2 - EndID + StartID ]])    
                            self.clause(CNF_data, variable_number, Aux, StartID + (block_number-2)*block_len, StartID + (block_number-1)*block_len -1, c[Aux-2], current_depth-1, depth)

                            CCX | self._cgate([c[Aux-2]  , c[Aux-1], c[ Aux-2 - EndID + StartID ]])     
                            self.clause(CNF_data, variable_number, Aux, StartID + (block_number-1)*block_len, EndID, c[Aux-1], current_depth-1, depth)

                            CCX | self._cgate([c[Aux-2]  , c[Aux-1], c[ Aux-2 - EndID + StartID ]])  
                            self.clause(CNF_data, variable_number, Aux, StartID + (block_number-2)*block_len, StartID + (block_number-1)*block_len -1, c[Aux-2], current_depth-1, depth)
                            
                            CCX | self._cgate([c[Aux-2]  , c[Aux-1], c[ Aux-2 - EndID + StartID ]])    
                                
                                #控制位variable_number + Aux - (block_number-1) + 2 -j 放 clause ， 另一个 控制位(将与此for内的前一个target 相同)并和target 依次上升
                            for j in range( block_number-3, 0 , -1):
                                #print(Aux, block_number,j,EndID,StartID, Aux-(block_number-1)+j-1,  Aux- 2*(block_number-1) + j ,Aux- 2*(block_number-1) -1 + j,len(c))
                                CCX | self._cgate([c[Aux-(block_number-1)+j-1] , c[Aux- 2*(block_number-1) + j], c[Aux- 2*(block_number-1) -1 + j]])
                                self.clause(CNF_data, variable_number, Aux, StartID + j*block_len , StartID -1 +(1+ j)*block_len, c[Aux-(block_number-1)+j-1], current_depth-1, depth)
                                CCX | self._cgate([c[Aux-(block_number-1)+j-1] , c[Aux- 2*(block_number-1) + j], c[Aux- 2*(block_number-1) -1 + j]])

                            CCX | self._cgate([c[Aux  -block_number ] , c[Aux - 2*(block_number-1)] , target])
                            self.clause(CNF_data, variable_number, Aux, StartID, StartID -1 + block_len , c[Aux  -block_number ], current_depth-1, depth)
                            CCX | self._cgate([c[Aux  -block_number] , c[Aux - 2*(block_number-1)] , target])

                                # 还原各个位置

                                #控制位variable_number + Aux - (block_number-1) + 2 -j 放 clause ， 另一个 控制位(将与此for内的前一个target 相同)并和target 依次上升
                            for j in range(1, block_number-1 - 1):
                                CCX | self._cgate([c[Aux-(block_number-1)+j-1] , c[Aux- 2*(block_number-1) + j], c[Aux- 2*(block_number-1) -1 + j]])
                                self.clause(CNF_data, variable_number, Aux, StartID + j*block_len , StartID -1 +(1+ j)*block_len, c[Aux-(block_number-1)+j-1], current_depth-1, depth)
                                CCX | self._cgate([c[Aux-(block_number-1)+j-1] , c[Aux- 2*(block_number-1) + j], c[Aux- 2*(block_number-1) -1 + j]])
                                    
                                # topPhase 
                            CCX | self._cgate([c[Aux-2]  , c[Aux-1], c[ Aux-2 - EndID + StartID ]])    
                            self.clause(CNF_data, variable_number, Aux, StartID + (block_number-2)*block_len, StartID + (block_number-1)*block_len -1, c[Aux-2], current_depth-1, depth)

                            CCX | self._cgate([c[Aux-2]  , c[Aux-1], c[ Aux-2 - EndID + StartID ]])     
                            self.clause(CNF_data, variable_number, Aux, StartID + (block_number-1)*block_len, EndID, c[Aux-1], current_depth-1, depth)

                            CCX | self._cgate([c[Aux-2]  , c[Aux-1], c[ Aux-2 - EndID + StartID ]])  
                            self.clause(CNF_data, variable_number, Aux, StartID + (block_number-2)*block_len, StartID + (block_number-1)*block_len -1, c[Aux-2], current_depth-1, depth)
                            
                            CCX | self._cgate([c[Aux-2]  , c[Aux-1], c[ Aux-2 - EndID + StartID ]])    
                                
                                #控制位variable_number + Aux - (block_number-1) + 2 -j 放 clause ， 另一个 控制位(将与此for内的前一个target 相同)并和target 依次上升
                            for j in range( block_number-3, 0 , -1):
                                CCX | self._cgate([c[Aux-(block_number-1)+j-1] , c[Aux- 2*(block_number-1) + j], c[Aux- 2*(block_number-1) -1 + j]])
                                self.clause(CNF_data, variable_number, Aux, StartID + j*block_len , StartID -1 +(1+ j)*block_len, c[Aux-(block_number-1)+j-1], current_depth-1, depth)
                                CCX | self._cgate([c[Aux-(block_number-1)+j-1] , c[Aux- 2*(block_number-1) + j], c[Aux- 2*(block_number-1) -1 + j]])


