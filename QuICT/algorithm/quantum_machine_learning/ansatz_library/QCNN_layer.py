import numpy as np
import math
import torch
from QuICT.core import Circuit
from QuICT.core.gate import H, CX,Rz,Ry,CU3,U3,CompositeGate,BasicGate
from QuICT.core.operator import Trigger
from QuICT.algorithm.quantum_machine_learning.ansatz_library.QNN_layer import QNNLayer
from QuICT.simulation.simulator import Simulator
from QuICT.algorithm.quantum_machine_learning.utils.hamiltonian import Hamiltonian
from QuICT.algorithm.quantum_machine_learning.encoding import NEQR,FRQI
from QuICT.core.utils.variable import Variable
from QuICT.algorithm.quantum_machine_learning.differentiator.parameter_shift import  ParameterShift


class QCNNLayer(QNNLayer):
    def __init__(self, params,data_qubits, result_qubit, device,):
        super().__init__(data_qubits, result_qubit, device)
        self.param = params
        """
      
        Args:
        params(Variable):  the number of elements in self.param is not exactly match the number of parameters 
        in model-circuit,and the former should be not less than the after.
     
        """
        
    def qconv(self,wires:list,idx_param:list):
        """
        The qconv2 class, which is a quatnum convoluitional used in QCNN model
        Args:
            wires(list): the wires where qcnov acting on
            idx_param(list): index of self.param ,it is expected to be a 1*4 list
        """
        kernal_gate = CompositeGate()
        param = self.param[idx_param]
        Ry(param[0]) | kernal_gate(wires[0])
        Ry(param[1]) | kernal_gate(wires[1])
        CX | kernal_gate([wires[1], wires[0]])
        Ry(param[2]) | kernal_gate(wires[0])
        Ry(param[3]) | kernal_gate(wires[1])
        CX | kernal_gate([wires[0], wires[1]])
        return kernal_gate
    
    def pool(self,wires:list,idx_param:list):
        """
        Args:
            wires(list): the index of qubits  which circuit_layer acting on 
            idx_param(list): contain the index of parameter ,it is expected to be a 1*6 list
            reference:
            http://www.juestc.uestc.edu.cn/cn/article/doi/10.12178/1001-0548.2022279
        """
        pool_gate = CompositeGate()  # mearsurement gate
        param = self.param[idx_param]

        Rz(param[0]) | pool_gate(wires[0])
        Ry(param[1]) | pool_gate(wires[0])
        Rz(param[2]) | pool_gate(wires[0])
        Rz(param[3]) | pool_gate(wires[1])
        Ry(param[4]) | pool_gate(wires[1])
        Rz(param[5]) | pool_gate(wires[1])
        CX | pool_gate([wires[0], wires[1]])
        Rz(param[3]).inverse() | pool_gate(wires[1])
        Ry(param[4]).inverse() | pool_gate(wires[1])
        Rz(param[5]).inverse() | pool_gate(wires[1])
            
        return pool_gate
    
    def FC(self, wires:list, idx1_param, idx2_param, )-> CompositeGate:
        """
        Args:
            wires(list): the index of qubits  which circuit_layer acting on 
            idx_param(list): contain the index of parameter 
            reference:
            https://journals.aps.org/pra/abstract/10.1103/PhysRevA.101.032308
        """
        G_param = self.param[idx1_param]  # the param of U3gates,it is expected to be a  Variable with _*3 size
        G2_param = self.param[idx2_param]  # the param of CU3gates,it is expected to be a  Variable with _*3 size
   
        fc_gate = CompositeGate()
        l= len(wires)
        for i in range(l):
            G_gate = U3
            G_gate = G_gate(G_param[3*i],G_param[3*i+1],G_param[3*i+2])
            G_gate & wires[i] | fc_gate
        for i in range(l):
            cu3_gate = CU3
            cu3_gate = cu3_gate(G2_param[3*i],G2_param[3*i+1],G2_param[3*i+2])
            cu3_gate.cargs=[wires[(l-i)%l]]
            cu3_gate.targs=[wires[l-i-1]]
            cu3_gate & [wires[(l-i)%l],wires[l-i-1]]|fc_gate  
            #cu3_gate |fc_gate  ([(l-i)%len(G2_param),l-i-1])
        return fc_gate

    def circuit_layer(self,idx_param:list,wires:list,com_gate:CompositeGate):
        """
        one layer contains two conv and one pool
        Args:
            idx_param(list): contain the index of parameter 
            list[0] contains params for qconv
            list[1] contains params for pool
            wires(list): the index of qubits  which circuit_layer acting on 
        return:
        com_gate(CompositeGate):composite gate of circuit layer
        """
        index_param0 = 0
        index_param1= 0
        new_wires=list()
        n_wires=len(wires)
        for i in range(1,n_wires,2): 
            wire_list1=[wires[i],wires[(i+1)%n_wires]]
            wire_list2=[wires[i-1],wires[(i)%n_wires]]
            self.qconv(wires=wire_list1,idx_param= idx_param[0][index_param0:index_param0+4]) |com_gate
            index_param0=index_param0+4
            self.qconv(wires=wire_list2,idx_param= idx_param[0][index_param0:index_param0+4]) |com_gate
            index_param0=index_param0+4
            self.pool(idx_param=idx_param[1][index_param1:index_param1+6],wires=wire_list2)|com_gate
            index_param1=index_param1+6
        
        for i in range(0,len(wires),2):
            new_wires.append(i)
        wires=new_wires
        
        return com_gate,wires
