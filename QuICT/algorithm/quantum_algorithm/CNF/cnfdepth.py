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
        CleanQubitNumber = math.floor((ancilla_qubits_num) / (S+1))   # The number of clean and dirty bits is+1=twice the number of blocks that can be processed by the non bottom layer
        assert CleanQubitNumber > 2 , "Need more auxiliary qubit for enough momery space."
        MemoryQubitNumber = ancilla_qubits_num - 2 * CleanQubitNumber 
        assert ancilla_qubits_num > 2 * clause_length + 2 , "Need more auxiliary qubit or some CNF-clause contains too many variables.."
        ClauseForFirstDep = 1 + math.floor( MemoryQubitNumber / clause_length) 
        p = math.floor((CleanQubitNumber + 1)/2)
        depth = math.ceil(math.log( math.ceil(clause_number / ClauseForFirstDep), p )) + 1
        target = variable_number
        if clause_number == 1:            #n= variable_number + Aux + 1
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

            for i in range(len(controls_X)):
                X | self._cgate(controls_X[i])
        else:
            if ((clause_number - 1 ) * (clause_length + 1 ) + 2 * clause_number + 1 ) < ancilla_qubits_num :
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
                
                #restore   
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

                    for i in range(len(controls_X)):
                        X | self._cgate(controls_X[i])
            else:

                if (clause_number < 1 + math.floor( (CleanQubitNumber + 1) /2)): 
                    controls_XX=[]
                    for j in range(1, clause_number + 1): #
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
                            MCTOneAux().execute(len(controls_abs) + 2) | self._cgate(controls_abs + [ variable_number + CleanQubitNumber + j  , variable_number + j ])
                        else:
                            X | self._cgate(variable_number + CleanQubitNumber + j)
                        for i in range(len(controls_X)):
                            X | self._cgate(controls_X[i])
                     
                    d = set(range(variable_number + 1 + ancilla_qubits_num))
                    d.remove(variable_number)
                    for j in controls_XX:
                        d.remove(j)
                    d = list(d)  
                    if controls_XX != []:
                        MCTLinearHalfDirtyAux().execute(len(controls_XX), 1 + variable_number + ancilla_qubits_num) | self._cgate(controls_XX + d + [variable_number] )
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
                            MCTOneAux().execute(len(controls_abs) + 2) | self._cgate(controls_abs + [ variable_number + CleanQubitNumber + j  , variable_number + j ])
                        else:
                            X | self._cgate(variable_number + CleanQubitNumber + j)

                        for i in range(len(controls_X)):
                            X | self._cgate(controls_X[i])                       
                else:
                    p = math.floor( (CleanQubitNumber + 1) /2)                   
                    depth = math.ceil(math.log( clause_number , p)) - 1
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
 
                    for j in range(block_number):
                        self.clause(
                            CNF_data, variable_number, ancilla_qubits_num, clause_length, CleanQubitNumber,
                            j * block_len +1, np.minimum( (j+1) * block_len, clause_number),
                            variable_number + CleanQubitNumber - p + 1 + j, depth-1, depth
                        )


    def read_CNF(self, cnf_file):        # file analysis
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
            CNF_data.append(int_new)  #
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

            for i in range(len(controls_X)):
                X | self._cgate(controls_X[i])
        else:  #
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
                        variable_Parallel_value[abs(CNF_data[i][j])-1] += 1
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
                if  (( 1 + clause_length * 2 ) * (Parallel_depth_max + qmemo)) > variable_number +  CleanQubitNumber: # mct has enough aux or not
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
                    #restore

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
                                MCTOneAux().execute(len(controls_abs0) + 2) | self._cgate( controls_abs0 + [  cl_postition + variable_number + CleanQubitNumber + 1 , c[cl_postition] ] )                            # one_dirty_aux(self._cgate, controls_abs0, target, current_Aux)

                            for i in range(len(controls_X0)):
                                X | self._cgate(controls_X0[i])
                            cl_postition += 1

                else: #AUX enough
                    Parallel_depth_list = [] 
                    for j in range(EndID - StartID + 1):
                        Parallel_depth_list.append([])

                    variable_Parallel_value_max = variable_Parallel_value[0]
                    for j in range(variable_number):
                        if variable_Parallel_value_max < variable_Parallel_value[j]: 
                            variable_Parallel_value_max = variable_Parallel_value[j]
                    
                    tt = variable_Parallel_value_max #tt After running the following while, the maximum number of parallel clauses will be reached each time. Finally, tt returns the maximum number of parallel layers at the bottom
                    tb = 1
                    ans = tt
                    while ( tb < tt  ): 
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

                    tt = ans

                    mappingList = []   #
                    mapping_variable =[] #Store which variables will be mapped to other locations
                    ta = 0 
                    tb = 0
                    for j in range(variable_number):
                        if variable_Parallel_value[j] > tt:
                            mapping_variable.append(j)
                            for jj in range(math.ceil( variable_Parallel_value[j] / tt)):
                                mappingList.append( [j, variable_number + 1 + 2 * CleanQubitNumber + ta + jj] )
                                
                            ta += math.ceil( variable_Parallel_value[j] / tt - 1)

                    for j in range(ta):
                        CX | self._cgate([ mappingList[j][0], mappingList[j][1]] )


                    variable_check_list = []
                    variable_Parallel_value = [0] *  (variable_number + 1 + Aux)
                    clause_Parallel_value = [1] * (EndID + 1 - StartID)
                    
                    CNF_data_update = []
                    for j in range(EndID + 1):
                        CNF_data_update.append(CNF_data[j])
                    
                    tt = variable_Parallel_value_max 
                    for i in range(StartID, EndID + 1):
                        for j in range(len(CNF_data[i])):
                            if ((abs(CNF_data[i][j])-1) not in variable_check_list): #
                                variable_check_list.append((abs(CNF_data[i][j])-1))
                                variable_Parallel_value[abs(CNF_data[i][j])-1] += 1
                                if (variable_Parallel_value[abs(CNF_data[i][j])-1] ) > clause_Parallel_value[i-StartID]: 
                                    clause_Parallel_value[i-StartID] = (variable_Parallel_value[abs(CNF_data[i][j])-1] % tt) 
                            else: #
                                if (abs(CNF_data[i][j])-1) not in mapping_variable: #
                                    variable_Parallel_value[abs(CNF_data[i][j])-1] += 1
                                    if (variable_Parallel_value[abs(CNF_data[i][j])-1] ) > clause_Parallel_value[i-StartID]: 
                                        clause_Parallel_value[i-StartID] = (variable_Parallel_value[abs(CNF_data[i][j])-1] % tt) 
                                else: #
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
                    
                        Parallel_depth_list[(clause_Parallel_value[i-StartID]-1)].append(i)

                    cl_position = 0

                    for k in range(tt): #bottom: how to put clause
                        a1 = range( variable_number + 1 + Aux )
                        Auxqubit = set( list(a1) )
                        Auxqubit.remove(target)
                        for jjk in range(variable_number + 1 + CleanQubitNumber , variable_number + 1 + len(Parallel_depth_list[k])  +  CleanQubitNumber ):
                            Auxqubit.remove(jjk)
                        #Each layer records the position of the assigned auxiliary bit.
                        for kk in range(len(Parallel_depth_list[k])):
                            clasueID = Parallel_depth_list[k][kk]
                            controls = CNF_data_update[clasueID]
                            for i in range(len(controls)):
                                if controls[i] < 0  and  ((-controls[i]-1) in Auxqubit):
                                    Auxqubit.remove(-controls[i]-1)
                                if controls[i] > 0 and  ((controls[i]-1) in Auxqubit):
                                    Auxqubit.remove(controls[i]-1)
                        Auxqubit1 = list(Auxqubit)    

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
                        
                    b =  cl_position 
                    if EndID - StartID == 1:
                        CCX | self._cgate([variable_number + 1 + CleanQubitNumber, variable_number + 1 + 1 + CleanQubitNumber, target])
                    else:
                        for j in range(b-1):
                            CCX | self._cgate([variable_number + 1 + 2 * j + CleanQubitNumber , variable_number + 2 + 2 * j + CleanQubitNumber, variable_number + 1 + cl_position  + CleanQubitNumber])
                            cl_position += 1
                        CCX | self._cgate([variable_number - 1 + cl_position + CleanQubitNumber, variable_number + cl_position + CleanQubitNumber, target])
                        for j in range(b-2, -1 ,-1):
                            CCX | self._cgate([variable_number + 1 + 2 * j + CleanQubitNumber , variable_number + 2 + 2 * j + CleanQubitNumber, variable_number + cl_position  + CleanQubitNumber])
                            cl_position -= 1
                    
                    cl_position = 0 #restore   Auxqubit = [ ] * tt
                    for k in range(tt): 
                        a1 = range( variable_number + 2 + Aux )
                        Auxqubit = set( list(a1) )
                        Auxqubit.remove(target)
                        for kk in range(variable_number + 1 + CleanQubitNumber , variable_number + 1 + 2 * CleanQubitNumber ):
                            Auxqubit.remove(kk)
              
                        for kk in range(len(Parallel_depth_list[k])):
                            clasueID = Parallel_depth_list[k][kk]
                            controls = CNF_data_update[clasueID]
                            for i in range(len(controls)):
                                if controls[i] < 0  and  ((-controls[i]-1) in Auxqubit):
                                    Auxqubit.remove(-controls[i]-1)
                                if controls[i] > 0 and  ((controls[i]-1) in Auxqubit):
                                    Auxqubit.remove(controls[i]-1)
                        Auxqubit1 = list(Auxqubit)    

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
                #EndID-StartID 
                block_len = math.ceil((EndID - StartID +1) /p)
                block_number = math.ceil((EndID - StartID + 1) / block_len )
                if block_number == 2:
                    if ((depth - current_depth) % 2) == 1: # odd  target :variable_number + ancilla_qubits_num - p + 1 + j
                        
                        CCX | self._cgate([variable_number + 1  , variable_number  , target])    
                        self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, StartID, StartID + block_len -1 , variable_number + 1, current_depth-1, depth)

                        CCX | self._cgate([variable_number + 1  , variable_number  , target])
                        self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, StartID + block_len, EndID, variable_number , current_depth-1, depth)
                        
                        CCX | self._cgate([variable_number + 1  , variable_number  , target])
                        self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, StartID, StartID + block_len -1, variable_number + 1, current_depth-1, depth)

                        CCX | self._cgate([variable_number + 1  , variable_number  , target]) 
                        self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, StartID + block_len, EndID, variable_number , current_depth-1, depth)
                    else: # even
                        
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

                        CCX | self._cgate([c[block_number-1] , c[2*(block_number-1)-1] , target])
                        self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, 
                                    StartID, StartID + block_len -1 , c[block_number-1], current_depth-1, depth)
                        CCX | self._cgate([c[block_number-1] , c[2*(block_number-1)-1] , target])

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

                    else: #even
                        CCX | self._cgate([c[CleanQubitNumber  -block_number ] , c[CleanQubitNumber - 2*(block_number-1)] , target])
                        self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, StartID, StartID -1 + block_len , c[CleanQubitNumber  -block_number ], current_depth-1, depth)
                        CCX | self._cgate([c[CleanQubitNumber  -block_number] , c[CleanQubitNumber - 2*(block_number-1)] , target])
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
                    
                        for j in range( block_number-3, 0 , -1):
                            CCX | self._cgate([c[CleanQubitNumber-(block_number-1)+j-1] , c[CleanQubitNumber- 2*(block_number-1) + j], c[CleanQubitNumber- 2*(block_number-1) -1 + j]])
                            self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, StartID + j*block_len , StartID -1 +(1+ j)*block_len, c[CleanQubitNumber-(block_number-1)+j-1], current_depth-1, depth)
                            CCX | self._cgate([c[CleanQubitNumber-(block_number-1)+j-1] , c[CleanQubitNumber- 2*(block_number-1) + j], c[CleanQubitNumber- 2*(block_number-1) -1 + j]])

                        CCX | self._cgate([c[CleanQubitNumber - block_number ] , c[CleanQubitNumber -  2*(block_number-1)] , target])
                        self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, StartID, StartID -1 + block_len , c[CleanQubitNumber - block_number ], current_depth-1, depth)
                        CCX | self._cgate([c[CleanQubitNumber - block_number] , c[CleanQubitNumber -  2*(block_number-1)] , target])

                            # restore
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

                        for j in range( block_number-3, 0 , -1):
                            CCX | self._cgate([c[CleanQubitNumber-(block_number-1)+j-1] , c[CleanQubitNumber- 2*(block_number-1) + j], c[CleanQubitNumber- 2*(block_number-1) -1 + j]])
                            self.clause(CNF_data, variable_number, Aux, clause_length, CleanQubitNumber, StartID + j*block_len , StartID -1 +(1+ j)*block_len, c[CleanQubitNumber-(block_number-1)+j-1], current_depth-1, depth)
                            CCX | self._cgate([c[CleanQubitNumber-(block_number-1)+j-1] , c[CleanQubitNumber- 2*(block_number-1) + j], c[CleanQubitNumber- 2*(block_number-1) -1 + j]])
