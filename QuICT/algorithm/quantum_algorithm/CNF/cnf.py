# -*- coding:utf8 -*-
# @TIME    : 2022/7/
# @Author  : 
# @File    : 
from builtins import print
import math
import numpy as np
#pyfrom sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from QuICT.core import *     #Circuit
from QuICT.core.gate import *
from QuICT.qcda.synthesis.mct import MCTOneAux
from QuICT.core.operator import Trigger
import logging

#from .._synthesis import Synthesis
#import random import logging from math import pi, gcd
import numpy as np
from fractions import Fraction
from typing import List, Tuple

#from QuICT.core import Circuit
#from QuICT.core.gate import *cc

# from QuICT.simulation.cpu_simulator import CircuitSimulator
from QuICT.qcda.synthesis.mct import one_dirty_aux
from QuICT.qcda.synthesis.mct.mct_linear_simulation import half_dirty_aux
from QuICT.qcda.optimization.commutative_optimization import *
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
        assert ancilla_qubits_num >= 3, "Need at least 3 auxiliary qubit."

        # Step 1: Read CNF File
        variable_nunmber, clause_number, CNF_data = self.read_CNF(cnf_file)

        # Step 2: Construct Circuit
        self._cgate = CompositeGate()
        p = math.floor(ancilla_qubits_num / 2) + 1
        depth=math.ceil(math.log( clause_number , p ))
        target = variable_nunmber
        if clause_number==1:
            #n= variable_nunmber + Aux + 1
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
            one_dirty_aux(self._cgate, controls_abs, target, ancilla_qubits_num) #QuICT.qcda.synthesis.mct.
            X | self._cgate(target)
            for i in range(len(controls_X)):
                X | self._cgate(controls_X[i])
        else:
            block_len = p ** (math.ceil(math.log(clause_number,p))-1)
            block_number = math.ceil(clause_number / block_len )
            controls = []
            #if (math.floor((depth - current_depth) % 2)) == 1:    #math.floor(math.log(EndID-StartID,p))
            for j in range(block_number):
                self.clause(
                    CNF_data, variable_nunmber,
                    ancilla_qubits_num, j * block_len +1, np.minimum( (j+1) * block_len, clause_number),
                    variable_nunmber + ancilla_qubits_num - p + 1 + j, depth-1, depth
                )
                controls.append(variable_nunmber + ancilla_qubits_num - p + 1 + j)

            current_Aux = variable_nunmber + 1 
            one_dirty_aux(self._cgate, controls, target, current_Aux)
            
            for j in range(block_number):
                self.clause(
                    CNF_data, variable_nunmber,
                    ancilla_qubits_num,  j * block_len +1, np.minimum( (j+1) * block_len, clause_number),
                    variable_nunmber + ancilla_qubits_num - p + 1 + j, depth-1, depth
                )

        # print(self._cgate)
        print(self._cgate.qasm())

        # Step 3: Simulator
        # simu_circuit = Circuit(self._cgate.width())
        # self._cgate | simu_circuit

        # self._simulator.run(simu_circuit)

    def read_CNF(self, cnf_file):
        # file analysis
        #输入一个文件，输出CNF格式的list
        variable_nunmber = 0
        clause_number = 0
        CNF_data = []
        f = open(cnf_file, 'r') 
        for line in f.readlines():
            new = line.strip().split()
            int_new=[]
            if new[0] == 'p':
                variable_nunmber = int(new[2])
                clause_number = int(new[3])
            else:
                for i in range(len(new)-1): #注意这里是否减1 要检查一下
                    int_new.append(int(new[i]))
            CNF_data.append(int_new)  #给各个Clause 编号0,1 ...m-1#
        print(CNF_data)
        f.close()

        return variable_nunmber, clause_number, CNF_data

    def clause(self, CNF_data: List, variable_nunmber: int, Aux: int, StartID: int, EndID: int, target: int, current_depth: int, depth: int):
        p = math.floor(Aux/2) + 1
        if StartID == EndID:
            #n= variable_nunmber + Aux + 1
            controls = CNF_data[StartID]
            if target < variable_nunmber + p:
                current_Aux = target + 1
            else:
                current_Aux = target - 1
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
            X | self._cgate(target)
            for i in range(len(controls_X)):
                X | self._cgate(controls_X[i])
        else: 
            if StartID +1 == EndID :
                if (math.floor((depth - current_depth) % 2)) == 1:
                    if (target != variable_nunmber + 1) and (target != variable_nunmber + 2):
                        CCX | self._cgate([variable_nunmber + 1  , variable_nunmber + 2 , target])    
                        self.clause(CNF_data, variable_nunmber, Aux, StartID,StartID, variable_nunmber + 1, current_depth-1, depth)

                        CCX | self._cgate([variable_nunmber + 1  , variable_nunmber + 2 , target])
                        self.clause(CNF_data, variable_nunmber, Aux, EndID, EndID, variable_nunmber + 2, current_depth-1, depth)
                        
                        CCX | self._cgate([variable_nunmber + 1  , variable_nunmber + 2 , target])
                        self.clause(CNF_data, variable_nunmber, Aux, StartID,StartID, variable_nunmber + 1, current_depth-1, depth)

                        CCX | self._cgate([variable_nunmber + 1  , variable_nunmber + 2 , target])
                    else: 
                        if (target == variable_nunmber + 1):
                            CCX | self._cgate([variable_nunmber + 3  , variable_nunmber + 2 , target])    
                            self.clause(CNF_data, variable_nunmber, Aux, StartID,StartID, variable_nunmber + 3, current_depth-1, depth)

                            CCX | self._cgate([variable_nunmber + 3  , variable_nunmber + 2 , target])
                            self.clause(CNF_data, variable_nunmber, Aux, EndID, EndID, variable_nunmber + 2, current_depth-1, depth)
                            
                            CCX | self._cgate([variable_nunmber + 3  , variable_nunmber + 2 , target])
                            self.clause(CNF_data, variable_nunmber, Aux, StartID,StartID, variable_nunmber + 3, current_depth-1, depth)

                            CCX | self._cgate([variable_nunmber + 3  , variable_nunmber + 2 , target])
                        else:
                            CCX | self._cgate([variable_nunmber + 1  , variable_nunmber + 3 , target])    
                            self.clause(CNF_data, variable_nunmber, Aux, StartID,StartID, variable_nunmber + 1, current_depth-1, depth)

                            CCX | self._cgate([variable_nunmber + 1  , variable_nunmber + 3 , target])
                            self.clause(CNF_data, variable_nunmber, Aux, EndID, EndID, variable_nunmber + 3, current_depth-1, depth)
                            
                            CCX | self._cgate([variable_nunmber + 1  , variable_nunmber + 3 , target])
                            self.clause(CNF_data, variable_nunmber, Aux, StartID,StartID, variable_nunmber + 1, current_depth-1, depth)

                            CCX | self._cgate([variable_nunmber + 1  , variable_nunmber + 3 , target])
                else:
                    if (target != variable_nunmber + Aux -1) and (target != variable_nunmber + Aux):
                        CCX | self._cgate([variable_nunmber + Aux -1  , variable_nunmber + Aux , target])   
                        self.clause(CNF_data, variable_nunmber, Aux, StartID,StartID, variable_nunmber + Aux -1, current_depth-1, depth)

                        CCX | self._cgate([variable_nunmber + Aux -1  , variable_nunmber + Aux , target])
                        self.clause(CNF_data, variable_nunmber, Aux, EndID, EndID, variable_nunmber + Aux, current_depth-1, depth)
                        
                        CCX | self._cgate([variable_nunmber + Aux -1  , variable_nunmber + Aux , target])
                        self.clause(CNF_data, variable_nunmber, Aux, StartID,StartID, variable_nunmber + Aux -1, current_depth-1, depth)

                        CCX | self._cgate([variable_nunmber + Aux -1  , variable_nunmber + Aux , target]) 
                    else:
                        if (target == variable_nunmber + Aux -1):
                            CCX | self._cgate([variable_nunmber + Aux -2  , variable_nunmber + Aux , target])   
                            self.clause(CNF_data, variable_nunmber, Aux, StartID,StartID, variable_nunmber + Aux -2, current_depth-1, depth)

                            CCX | self._cgate([variable_nunmber + Aux -2  , variable_nunmber + Aux , target])
                            self.clause(CNF_data, variable_nunmber, Aux, EndID, EndID, variable_nunmber + Aux, current_depth-1, depth)
                            
                            CCX | self._cgate([variable_nunmber + Aux -2  , variable_nunmber + Aux , target])
                            self.clause(CNF_data, variable_nunmber, Aux, StartID,StartID, variable_nunmber + Aux -2, current_depth-1, depth)

                            CCX | self._cgate([variable_nunmber + Aux -2  , variable_nunmber + Aux , target]) 
                    
                   
            else:
                #print(EndID-StartID)
                #print(math.log(EndID-StartID, p))
                #print(math.ceil(math.log(EndID-StartID, p))-1)
                #print(p)
                block_len = p ** (math.ceil(math.log(EndID-StartID, p))-1)
                block_number = math.ceil((EndID-StartID) / block_len )
                controls = []
                #block_end = np.minimum(StartID + block_len-1, EndID)
                if (math.floor((depth - current_depth) % 2)) != 1:    #math.floor(math.log(EndID-StartID,p))
                    #层数差 奇数 的存储位 为 variable_nunmber +Aux- p+  至 variable_nunmber + Aux  要从差为偶数层 取数据
                    #层数差 偶数 的存储位 为 variable_nunmber 至 variable_nunmber + p -1      要从差为奇数层 取数据
                    #if block_number == 1:
                    #if p==2
                    #if block_number == 2:
                    
                    # UpPhase 1 升阶段 第一位要单独处理，其target 即最终target。控制位一个在variable_nunmber + Aux；另一个在 variable_nunmber + Aux - block_number +2 上， variable_nunmber + Aux - block_number +2 上放一个低一层的 clause。
                    CCX | self._cgate([variable_nunmber + Aux - block_number +2, variable_nunmber + Aux, target])
                    self.clause(CNF_data, variable_nunmber, Aux, StartID, np.minimum(StartID + block_len-1, EndID), variable_nunmber + Aux - block_number +2, current_depth-1, depth)
                    CCX | self._cgate([variable_nunmber + Aux - block_number +2, variable_nunmber + Aux , target])
                    
                    #控制位variable_nunmber + Aux - block_number + 2 -j 放 clause ， 另一个 控制位(将与此for内的前一个target 相同)并和target 依次上升
                    for j in range(1, block_number-2):
                        CCX | self._cgate([variable_nunmber + Aux - block_number + 2 -j , variable_nunmber + Aux -j, variable_nunmber + Aux +1-j])
                        self.clause(CNF_data, variable_nunmber, Aux, StartID + j * block_len, np.minimum(StartID + j * block_len-1, EndID), variable_nunmber + Aux - block_number + 2 -j, current_depth-1, depth)
                        CCX | self._cgate([variable_nunmber + Aux - block_number + 2 -j , variable_nunmber + Aux -j, variable_nunmber + Aux +1-j])
                            
                    # topPhase 

                    CCX | self._cgate([variable_nunmber + Aux - 2*block_number + 4  , variable_nunmber + Aux - 2*block_number + 3, variable_nunmber + Aux - block_number + 3])    
                    self.clause(CNF_data, variable_nunmber, Aux, StartID + (block_number-2) * block_len, np.minimum(StartID + (block_number-1) * block_len-1, EndID), variable_nunmber + Aux - 2*block_number + 4, current_depth-1, depth)

                    CCX | self._cgate([variable_nunmber + Aux - 2*block_number + 4  , variable_nunmber + Aux - 2*block_number + 3, variable_nunmber + Aux - block_number + 3])
                    self.clause(CNF_data, variable_nunmber, Aux, StartID + (block_number-1) * block_len, np.minimum(StartID + (block_number) * block_len-1, EndID), variable_nunmber + Aux - 2*block_number + 3, current_depth-1, depth)

                    CCX | self._cgate([variable_nunmber + Aux - 2*block_number + 4  , variable_nunmber + Aux - 2*block_number + 3, variable_nunmber + Aux - block_number + 3])
                    self.clause(CNF_data, variable_nunmber, Aux, StartID + (block_number-2) * block_len, np.minimum(StartID + (block_number-1) * block_len-1, EndID), variable_nunmber + Aux - 2*block_number + 4, current_depth-1, depth)

                    CCX | self._cgate([variable_nunmber + Aux - 2*block_number + 4  , variable_nunmber + Aux - 2*block_number + 3, variable_nunmber + Aux - block_number + 3])    
                
                    #QuICT.qcda.synthesis.mct.

                    #downPhase
                    for j in range(1, block_number-2):
                        jdown = block_number-2 - j
                        CCX | self._cgate([variable_nunmber + Aux - block_number + 2 - jdown, variable_nunmber + Aux -jdown, variable_nunmber + Aux +1-jdown] )
                        self.clause(CNF_data, variable_nunmber, Aux, StartID + jdown * block_len, np.minimum(StartID + jdown * block_len-1, EndID), variable_nunmber + Aux - block_number + 2 -jdown, current_depth-1, depth)
                        CCX | self._cgate([variable_nunmber + Aux - block_number + 2 -jdown , variable_nunmber + Aux -jdown, variable_nunmber + Aux +1-jdown] )
                    
                    CCX | self._cgate([variable_nunmber + Aux - block_number +2, variable_nunmber + Aux , target])
                    self.clause(CNF_data, variable_nunmber, Aux, StartID, np.minimum(StartID + block_len-1, EndID), variable_nunmber + Aux - block_number +2, current_depth-1, depth)
                    CCX | self._cgate([variable_nunmber + Aux - block_number +2, variable_nunmber + Aux , target])

                    #repeat....

                    # 还原各个位置

                    #控制位variable_nunmber + Aux - block_number + 2 -j 放 clause ， 另一个 控制位(将与此for内的前一个target 相同)并和target 依次上升
                    for j in range(1, block_number-2):
                        CCX | self._cgate([variable_nunmber + Aux - block_number + 2 -j , variable_nunmber + Aux -j, variable_nunmber + Aux +1-j] )
                        self.clause(CNF_data, variable_nunmber, Aux, StartID + j * block_len, np.minimum(StartID + j * block_len-1, EndID), variable_nunmber + Aux - block_number + 2 -j, current_depth-1, depth)
                        CCX | self._cgate([variable_nunmber + Aux - block_number + 2 -j , variable_nunmber + Aux -j, variable_nunmber + Aux +1-j])
                            
                    # topPhase 

                    CCX | self._cgate([variable_nunmber + Aux - 2*block_number + 4  , variable_nunmber + Aux - 2*block_number + 3, variable_nunmber + Aux - block_number + 3])    
                    self.clause(CNF_data, variable_nunmber, Aux, StartID + (block_number-2) * block_len, np.minimum(StartID + (block_number-1) * block_len-1, EndID), variable_nunmber + Aux - 2*block_number + 4, current_depth-1, depth)

                    CCX | self._cgate([variable_nunmber + Aux - 2*block_number + 4  , variable_nunmber + Aux - 2*block_number + 3, variable_nunmber + Aux - block_number + 3])
                    self.clause(CNF_data, variable_nunmber, Aux, StartID + (block_number-1) * block_len, np.minimum(StartID + (block_number) * block_len-1, EndID), variable_nunmber + Aux - 2*block_number + 3, current_depth-1, depth)

                    CCX | self._cgate([variable_nunmber + Aux - 2*block_number + 4  , variable_nunmber + Aux - 2*block_number + 3, variable_nunmber + Aux - block_number + 3])
                    self.clause(CNF_data, variable_nunmber, Aux, StartID + (block_number-2) * block_len, np.minimum(StartID + (block_number-1) * block_len-1, EndID), variable_nunmber + Aux - 2*block_number + 4, current_depth-1, depth)

                    CCX | self._cgate([variable_nunmber + Aux - 2*block_number + 4  , variable_nunmber + Aux - 2*block_number + 3, variable_nunmber + Aux - block_number + 3])    
                
                    #downPhase
                    for j in range(1, block_number-2):
                        jdown = block_number-2 - j
                        CCX | self._cgate([variable_nunmber + Aux - block_number + 2 - jdown, variable_nunmber + Aux -jdown, variable_nunmber + Aux +1-jdown] )
                        self.clause(CNF_data, variable_nunmber, Aux, StartID + jdown * block_len, np.minimum(StartID + jdown * block_len-1, EndID), variable_nunmber + Aux - block_number + 2 -jdown, current_depth-1, depth)
                        CCX | self._cgate([variable_nunmber + Aux - block_number + 2 -jdown , variable_nunmber + Aux -jdown, variable_nunmber + Aux +1-jdown] )
                    
                    # for j in range(block_number):
                    #    gate | Clause(CNF_data, variable_nunmber, Aux, StartID + j * block_len, np.minimum(StartID + j * block_len-1, EndID), variable_nunmber + j, current_depth-1, depth)
                else: #偶数层 处理
                    CCX | self._cgate([variable_nunmber + p-1 - block_number +2, variable_nunmber + p-1 , target])
                    self.clause( CNF_data, variable_nunmber, Aux, StartID, np.minimum(StartID + block_len-1, EndID), variable_nunmber + p-1 - block_number +2, current_depth-1, depth)
                    CCX | self._cgate([variable_nunmber + p-1 - block_number +2, variable_nunmber + p-1 , target])
                    
                    #控制位variable_nunmber + Aux - block_number + 2 -j 放 clause ， 另一个 控制位(将与此for内的前一个target 相同)并和target 依次上升
                    for j in range(1, block_number-2):
                        CCX | self._cgate([variable_nunmber + p-1 - block_number + 2 -j , variable_nunmber + p-1 -j, variable_nunmber + p-1 +1-j] )
                        self.clause(CNF_data, variable_nunmber, Aux, StartID + j * block_len, np.minimum(StartID + j * block_len-1, EndID), variable_nunmber + p-1 - block_number + 2 -j, current_depth-1, depth)
                        CCX | self._cgate([variable_nunmber + p-1 - block_number + 2 -j , variable_nunmber + p-1 -j, variable_nunmber + p-1 +1-j] )
                            
                    # topPhase 

                    CCX | self._cgate([variable_nunmber + p-1 - 2*block_number + 4  , variable_nunmber + p-1 - 2*block_number + 3, variable_nunmber + p-1 - block_number + 3])    
                    self.clause(CNF_data, variable_nunmber, Aux, StartID + (block_number-2) * block_len, np.minimum(StartID + (block_number-1) * block_len-1, EndID), variable_nunmber + p-1 - 2*block_number + 4, current_depth-1, depth)

                    CCX | self._cgate([variable_nunmber + p-1 - 2*block_number + 4  , variable_nunmber + p-1 - 2*block_number + 3, variable_nunmber + p-1 - block_number + 3])
                    self.clause(CNF_data, variable_nunmber, Aux, StartID + (block_number-1) * block_len, np.minimum(StartID + (block_number) * block_len-1, EndID), variable_nunmber + p-1 - 2*block_number + 3, current_depth-1, depth)

                    CCX | self._cgate([variable_nunmber + p-1 - 2*block_number + 4  , variable_nunmber + p-1 - 2*block_number + 3, variable_nunmber + p-1 - block_number + 3])
                    self.clause(CNF_data, variable_nunmber, Aux, StartID + (block_number-2) * block_len, np.minimum(StartID + (block_number-1) * block_len-1, EndID), variable_nunmber + p-1 - 2*block_number + 4, current_depth-1, depth)

                    CCX | self._cgate([variable_nunmber + p-1 - 2*block_number + 4  , variable_nunmber + p-1 - 2*block_number + 3, variable_nunmber + p-1 - block_number + 3])    
                
                    #QuICT.qcda.synthesis.mct.            

                    #downPhase
                    for j in range(1, block_number-2):
                        jdown = block_number-2 - j
                        CCX | self._cgate([variable_nunmber + p-1 - block_number + 2 - jdown, variable_nunmber + p-1 -jdown, variable_nunmber + p-1 +1-jdown] )
                        self.clause(CNF_data, variable_nunmber, Aux, StartID + jdown * block_len, np.minimum(StartID + jdown * block_len-1, EndID), variable_nunmber + p-1 - block_number + 2 -jdown, current_depth-1, depth)
                        CCX | self._cgate([variable_nunmber + p-1 - block_number + 2 -jdown , variable_nunmber + p-1 -jdown, variable_nunmber + p-1 +1-jdown] )
                    
                    CCX | self._cgate([variable_nunmber + p-1 - block_number +2, variable_nunmber + p-1 , target])
                    self.clause(CNF_data, variable_nunmber, Aux, StartID, np.minimum(StartID + block_len-1, EndID), variable_nunmber + p-1 - block_number +2, current_depth-1, depth)
                    CCX | self._cgate([variable_nunmber + p-1 - block_number +2, variable_nunmber + p-1 , target])

                    #repeat....

                    # 还原各个位置

                    #控制位variable_nunmber + Aux - block_number + 2 -j 放 clause ， 另一个 控制位(将与此for内的前一个target 相同)并和target 依次上升
                    for j in range(1, block_number-2):
                        CCX | self._cgate([variable_nunmber + p-1 - block_number + 2 -j , variable_nunmber + p-1 -j, variable_nunmber + p-1 +1-j] )
                        self.clause(CNF_data, variable_nunmber, Aux, StartID + j * block_len, np.minimum(StartID + j * block_len-1, EndID), variable_nunmber + p-1 - block_number + 2 -j, current_depth-1, depth)
                        CCX | self._cgate([variable_nunmber + p-1 - block_number + 2 -j , variable_nunmber + p-1 -j, variable_nunmber + p-1 +1-j] )
                            
                    # topPhase 

                    CCX | self._cgate([variable_nunmber + p-1 - 2*block_number + 4  , variable_nunmber + p-1 - 2*block_number + 3, variable_nunmber + p-1 - block_number + 3])    
                    self.clause(CNF_data, variable_nunmber, Aux, StartID + (block_number-2) * block_len, np.minimum(StartID + (block_number-1) * block_len-1, EndID), variable_nunmber + p-1 - 2*block_number + 4, current_depth-1, depth)

                    CCX | self._cgate([variable_nunmber + p-1 - 2*block_number + 4  , variable_nunmber + p-1 - 2*block_number + 3, variable_nunmber + p-1 - block_number + 3])
                    self.clause(CNF_data, variable_nunmber, Aux, StartID + (block_number-1) * block_len, np.minimum(StartID + (block_number) * block_len-1, EndID), variable_nunmber + p-1 - 2*block_number + 3, current_depth-1, depth)

                    CCX | self._cgate([variable_nunmber + p-1 - 2*block_number + 4  , variable_nunmber + p-1 - 2*block_number + 3, variable_nunmber + p-1 - block_number + 3])
                    self.clause(CNF_data, variable_nunmber, Aux, StartID + (block_number-2) * block_len, np.minimum(StartID + (block_number-1) * block_len-1, EndID), variable_nunmber + p-1 - 2*block_number + 4, current_depth-1, depth)

                    CCX | self._cgate([variable_nunmber + p-1 - 2*block_number + 4  , variable_nunmber + p-1 - 2*block_number + 3, variable_nunmber + p-1 - block_number + 3])    
                
                    #QuICT.qcda.synthesis.mct.
             
                    #downPhase
                    for j in range(1, block_number-2):
                        jdown = block_number-2 - j
                        CCX | self._cgate([variable_nunmber + p-1 - block_number + 2 - jdown, variable_nunmber + p-1 -jdown, variable_nunmber + p-1 +1-jdown] )
                        self.clause(CNF_data, variable_nunmber, Aux, StartID + jdown * block_len, np.minimum(StartID + jdown * block_len-1, EndID), variable_nunmber + p-1 - block_number + 2 -jdown, current_depth-1, depth)
                        CCX | self._cgate([variable_nunmber + p-1 - block_number + 2 -jdown , variable_nunmber + p-1 -jdown, variable_nunmber + p-1 +1-jdown] )
    
        

if __name__=="__main__":
    cnf = CNFSATOracle()
    cnf.run("./1.cnf") 
    #./QuICT/algorithm/quantum_algorithm/CNF/
    cgate = cnf.circuit()
    
    #print(cgate.qasm())
    circuit_temp=Circuit(15)
    circuit_temp.extend(cgate)
    circuit_temp.draw(filename='1.jpg')


# python QuICT/algorithm/qm/cnf/cnf.py """