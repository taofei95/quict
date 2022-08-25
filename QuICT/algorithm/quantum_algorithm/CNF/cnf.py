#
# -*- coding:utf8 -*-
# @TIME    : 2022/7/
# @Author  : 
# @File    : 
from builtins import print
from ast import If
import math
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from QuICT.core import *     #Circuit
from QuICT.core.gate import *
from QuICT.qcda.synthesis.mct import MCTOneAux
from QuICT.core.operator import Trigger
import logging

#from .._synthesis import Synthesis
import random
#import logging
from math import pi, gcd
import numpy as np
from fractions import Fraction
from typing import List, Tuple

#from QuICT.core import Circuit
#from QuICT.core.gate import *

# from QuICT.simulation.cpu_simulator import CircuitSimulator
from QuICT.qcda.synthesis.mct import one_dirty_aux
from QuICT.qcda.synthesis.mct.mct_linear_simulation import half_dirty_aux
from QuICT.qcda.optimization.commutative_optimization import CommutativeOptimization

class CNFSATOracle:
    # 
    def __init__(self, simu = None):
        self.simulator = simu

    def circuit(self) -> CompositeGate:
        """construct CNF-SAT oracle circuit
        Returns:
            Circuit: 
            List[int]: the indices to be measured to get ~phi
        """
        return self._cgates

    def execute(m, n):
        gates = CompositeGate()
        #with gates:
            
        #return gates
    
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
        if clause_number==1:
            #n= variable_nunmber + Aux + 1
            controls = CNF_data[0]
            target = variable_nunmber
            current_Aux = target + 1
            one_dirty_aux(controls, target, current_Aux) #QuICT.qcda.synthesis.mct.
        else:
            block_len = p ** (math.ceil(math.log(clause_number,p))-1)
            block_number = math.ceil((clause_number) / block_len )
            controls = []
            #if (math.floor((depth - current_depth) % 2)) == 1:    #math.floor(math.log(EndID-StartID,p))
            for j in range(block_number):
                self.clause(
                    CNF_data, variable_nunmber,
                    ancilla_qubits_num, j * block_len, np.minimum( (j+1) * block_len-1, clause_number),
                    variable_nunmber +p+ j, depth-1, depth
                )
                #(j - 1)s+1, js, Ancilla[j], d-1) ?new[j]=j
                controls.append(variable_nunmber +p+ j)
            current_Aux = variable_nunmber + 1 
            self._cgate | one_dirty_aux(self._ccontrols, variable_nunmber, current_Aux)
            for j in range(block_number):
                self.clause(
                    CNF_data, variable_nunmber,
                    ancilla_qubits_num,  j * block_len, np.minimum((j+1) * block_len-1, clause_number),
                    variable_nunmber +p+ j, depth-1, depth
                )

        print(self._cgate)
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
            if new[0] == 'p':
                variable_nunmber = int(new[2])
                clause_number = int(new[3])
                continue
            else:
                for i in range(len(new)-1): #注意这里是否减1 要检查一下
                    new[i] = int(new[i]) - 1
            CNF_data.append(new)  #给各个Clause 编号0,1 ...m-1#
        #for i in range(len(CNF_data)):    k = max(k,len(CNF_data[i]) - 1)
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
                one_dirty_aux(self._ccontrols, target, current_Aux)
        else:
            print(EndID-StartID)
            print(math.log(EndID-StartID, p))
            print(math.ceil(math.log(EndID-StartID, p))-1)
            print(p)
            block_len = p ** (math.ceil(math.log(EndID-StartID, p))-1)
            block_number = math.ceil((EndID-StartID) / block_len )
            controls = []
            #block_end = np.minimum(StartID + block_len-1, EndID)
            if (math.floor((depth - current_depth) % 2)) == 1:    #math.floor(math.log(EndID-StartID,p))
                #层数差 奇数 的存储位 为 variable_nunmber + p  至 variable_nunmber + Aux  要从差为偶数层 取数据
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

                CCX | self._cgate([[variable_nunmber + Aux - 2*block_number + 4  , variable_nunmber + Aux - 2*block_number + 3, variable_nunmber + Aux - block_number + 3]])    
                self.clause(CNF_data, variable_nunmber, Aux, StartID + (block_number-2) * block_len, np.minimum(StartID + (block_number-1) * block_len-1, EndID), variable_nunmber + Aux - 2*block_number + 4, current_depth-1, depth)

                CCX | self._cgate([[variable_nunmber + Aux - 2*block_number + 4  , variable_nunmber + Aux - 2*block_number + 3, variable_nunmber + Aux - block_number + 3]])
                self.clause(CNF_data, variable_nunmber, Aux, StartID + (block_number-1) * block_len, np.minimum(StartID + (block_number) * block_len-1, EndID), variable_nunmber + Aux - 2*block_number + 3, current_depth-1, depth)

                CCX | self._cgate([[variable_nunmber + Aux - 2*block_number + 4  , variable_nunmber + Aux - 2*block_number + 3, variable_nunmber + Aux - block_number + 3]])
                self.clause(CNF_data, variable_nunmber, Aux, StartID + (block_number-2) * block_len, np.minimum(StartID + (block_number-1) * block_len-1, EndID), variable_nunmber + Aux - 2*block_number + 4, current_depth-1, depth)

                CCX | self._cgate([[variable_nunmber + Aux - 2*block_number + 4  , variable_nunmber + Aux - 2*block_number + 3, variable_nunmber + Aux - block_number + 3]])    
               
                #QuICT.qcda.synthesis.mct.
                #gate | one_dirty_aux(gates, controls, target, current_Aux)
             

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
            else: 
                CCX | self._cgate([variable_nunmber + p-1 - block_number +2, variable_nunmber + p-1 , target])
                self.clause( CNF_data, variable_nunmber, Aux, StartID, np.minimum(StartID + block_len-1, EndID), variable_nunmber + p-1 - block_number +2, current_depth-1, depth)
                CCX | self._cgate([variable_nunmber + p-1 - block_number +2, variable_nunmber + p-1 , target])
                
                #控制位variable_nunmber + Aux - block_number + 2 -j 放 clause ， 另一个 控制位(将与此for内的前一个target 相同)并和target 依次上升
                for j in range(1, block_number-2):
                    CCX | self._cgate([[variable_nunmber + p-1 - block_number + 2 -j , variable_nunmber + p-1 -j, variable_nunmber + p-1 +1-j]] )
                    self.clause(CNF_data, variable_nunmber, Aux, StartID + j * block_len, np.minimum(StartID + j * block_len-1, EndID), variable_nunmber + p-1 - block_number + 2 -j, current_depth-1, depth)
                    CCX | self._cgate([[variable_nunmber + p-1 - block_number + 2 -j , variable_nunmber + p-1 -j, variable_nunmber + p-1 +1-j]] )
                         
                # topPhase 

                CCX | self._cgate([variable_nunmber + p-1 - 2*block_number + 4  , variable_nunmber + p-1 - 2*block_number + 3, variable_nunmber + p-1 - block_number + 3])    
                self.clause(CNF_data, variable_nunmber, Aux, StartID + (block_number-2) * block_len, np.minimum(StartID + (block_number-1) * block_len-1, EndID), variable_nunmber + p-1 - 2*block_number + 4, current_depth-1, depth)

                CCX | self._cgate([variable_nunmber + p-1 - 2*block_number + 4  , variable_nunmber + p-1 - 2*block_number + 3, variable_nunmber + p-1 - block_number + 3])
                self.clause(CNF_data, variable_nunmber, Aux, StartID + (block_number-1) * block_len, np.minimum(StartID + (block_number) * block_len-1, EndID), variable_nunmber + p-1 - 2*block_number + 3, current_depth-1, depth)

                CCX | self._cgate([variable_nunmber + p-1 - 2*block_number + 4  , variable_nunmber + p-1 - 2*block_number + 3, variable_nunmber + p-1 - block_number + 3])
                self.clause(CNF_data, variable_nunmber, Aux, StartID + (block_number-2) * block_len, np.minimum(StartID + (block_number-1) * block_len-1, EndID), variable_nunmber + p-1 - 2*block_number + 4, current_depth-1, depth)

                CCX | self._cgate([variable_nunmber + p-1 - 2*block_number + 4  , variable_nunmber + p-1 - 2*block_number + 3, variable_nunmber + p-1 - block_number + 3])    
               
                #QuICT.qcda.synthesis.mct.
                #gate | one_dirty_aux(gates, controls, target, current_Aux)
             

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
                #gate | one_dirty_aux(gates, controls, target, current_Aux)
             

                #downPhase
                for j in range(1, block_number-2):
                    jdown = block_number-2 - j
                    CCX | self._cgate([variable_nunmber + p-1 - block_number + 2 - jdown, variable_nunmber + p-1 -jdown, variable_nunmber + p-1 +1-jdown] )
                    self.clause(CNF_data, variable_nunmber, Aux, StartID + jdown * block_len, np.minimum(StartID + jdown * block_len-1, EndID), variable_nunmber + p-1 - block_number + 2 -jdown, current_depth-1, depth)
                    CCX | self._cgate([variable_nunmber + p-1 - block_number + 2 -jdown , variable_nunmber + p-1 -jdown, variable_nunmber + p-1 +1-jdown] )
 
    
# """    def Gand(CNF_data: List, variable_nunmber: int, Aux: int, StartID: int, EndID: int, target: int, current_depth: int, depth: int):
        
#         UpPhase(CNF_data: List, variable_nunmber: int, Aux: int, StartID: int, EndID: int, target: int, current_depth-1, depth)
               
#         TopPhase(CNF_data: List, variable_nunmber: int, Aux: int, StartID: int, EndID: int, target: int, current_depth-1, depth)
                
#         DownPhase(CNF_data: List, variable_nunmber: int, Aux: int, StartID: int, EndID: int, target: int, current_depth-1, depth)
            
#         #DownPhase(CNF_data: List, variable_nunmber: int, Aux: int, StartID: int, EndID: int, target: int, current_depth-1, depth)



#     def UpPhase(CNF_data: List, variable_nunmber: int, Aux: int, StartID: int, EndID: int, target: int, current_depth-1, depth):
#         p = math.floor(Aux/2) + 1
#         if (math.floor((depth - current_depth) % 2)) == 1：
#             for k in [1,p]:
#                 CCX &  | Clause[ , ]
#                 CCX &  | Clause[ , ]
#         else: h

#     def TopPhase(CNF_data: List, variable_nunmber: int, Aux: int, StartID: int, EndID: int, target: int, current_depth-1, depth):
#         CCX &
#         CCX &

#     def DownPhase(CNF_data: List, variable_nunmber: int, Aux: int, StartID: int, EndID: int, target: int, current_depth-1, depth):
#         for k in [,]
#             123

#    def Clause(StartID, EndID, l, Target, depth):
#         if StartID == EndID:
#             One_clause(StartID)
#         else:
#             for 123 123 123c 3sfs2d 
#             continue
#     def dirty_aux(controls : list , target: int)：
#     return

#     def clause_list_generation(CNF_data):
#     #generate the clause list, each item contain the size and CNOT count of circuit that implement the clause and the repeat times
#     #传入 CNF_data, 生成 clause_list
#     clause_list = []
#     for i in range(len(CNF_data)-1):
#         len(CNF_data[i])-2
#         Xcount = 1
#         for j in range(len(CNF_data[i])):
            
#         new = []
#         new.append(temp[0] + Xcount)
#         new.append(temp[1])
#         new.append(0)
#         clause_list.append(new)


# if __name__=="__main__":
#     cnf = CNFSATOracle()
#     cnf.run(file_path)
#     cgate = cnf.circuit()
#     print(cgate.qasm())


# python QuICT/algorithm/qm/cnf/cnf.py """
