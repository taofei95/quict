import numpy as np
import math
import torch
from QuICT.core import Circuit
from QuICT.core.gate import H, CX,Rz,UnitaryGate,CU3Gate,U3Gate,CompositeGate,BasicGate,RyGate
from QuICT_ml.utils.gate_tensor import Ry,CX_tensor
from QuICT.core.operator import Trigger
from QuICT.algorithm.quantum_machine_learning.utils.ansatz import Ansatz
from QuICT.algorithm.quantum_machine_learning.ansatz_library.QNN_layer import QNNLayer
from QuICT.simulation.simulator import Simulator
from QuICT.algorithm.quantum_machine_learning.utils.hamiltonian import Hamiltonian
from QuICT.algorithm.quantum_machine_learning.utils.encoding import NEQR,FRQI
from QuICT.algorithm.quantum_machine_learning.differentiators.parameter_shift import  ParameterShift
#from QuICT.algorithm.quantum_machine_learning.model.VQA import vqe_net
class qconv2:
    """The qconv2 class, which is a quatnum convoluitional used in QCNN model"""
    def __init__(self,) -> None:
        """
        Args:
            
        """
        
    def __call__(self, _type: str, wires:list,param) -> CompositeGate:
        """
        Args:
            param(list): the parm of gates,it is expected to be a 1*4 list if _type == "0"
        """
        self.param = param
        kernal_gate =CompositeGate()
        if _type == "0":
            Ry(float(self.param[0]))|kernal_gate(wires[0])
            Ry(float(self.param[1]),)|kernal_gate(wires[1])
            CX | kernal_gate([wires[1] ,wires[0]])
            Ry(float(self.param[2]),)|kernal_gate(wires[0])
            Ry(float(self.param[3]),)|kernal_gate(wires[1])
            CX | kernal_gate([wires[0],wires[1]])
        for gate in kernal_gate.gates:
            if isinstance(gate,RyGate):
                gate._requires_grad =True   
        return kernal_gate
    def _construct_ansatz(self,param):
        self.param = param
        ansatz = Ansatz(2,)


class pool:
    def __init__(self,) -> None:
        """
        Args:
        """
        
    def __call__(self, _type: str,param:list,wires:list) -> Trigger:
        """
        Args:
            parm(list): the parm of gates,it is expected to be a 1*6 list
            reference:
            http://www.juestc.uestc.edu.cn/cn/article/doi/10.12178/1001-0548.2022279
        """
        self.param = param
        pool_gate = CompositeGate()  # mearsurement gate
        if _type == "0":
            Rz(float(self.param[0]))|pool_gate(wires[0])
            Ry(float(self.param[1]))|pool_gate(wires[0])
            Rz(float(self.param[2]))|pool_gate(wires[0])
            Rz(float(self.param[3]))|pool_gate(wires[1])
            Ry(float(self.param[4]))|pool_gate(wires[1])
            Rz(float(self.param[5]))|pool_gate(wires[1])
            CX|pool_gate([wires[0],wires[1]])
            Rz(float(self.param[3])).inverse() |pool_gate(wires[1])
            Ry(float(self.param[4])).inverse() |pool_gate(wires[1])
            Rz(float(self.param[5])).inverse() |pool_gate(wires[1])
        for gate in pool_gate.gates:
            if gate.type ==CX().type:
                continue
            gate._requires_grad = True
            
        return pool_gate
class FC:
    def __init__(self,n_qubits:int,range_control = 1) -> None:
        """

        Args:
            param(list): the param of gates,it is expected to be a  list
            range_control: , where r is the 'range' of the control 
            reference:
            https://journals.aps.org/pra/abstract/10.1103/PhysRevA.101.032308
        """
        self.n_qubits=n_qubits
        self.r = range_control
    def __call__(self,G_param:list,G2_param:list,wires:list)-> CompositeGate:
        """
        Args:
            G_param(list): the param of U3gates,it is expected to be a  list with _*3 size
            G2_param(list): the param of CU3gates,it is expected to be a  list with _*3 size
            reference:
            https://journals.aps.org/pra/abstract/10.1103/PhysRevA.101.032308
        """
        fc_gate = CompositeGate()
        for i in range(len(wires)):
            G_gate = U3Gate()
            G_gate = G_gate(G_param[i][0],G_param[i][1],G_param[i][2])
            G_gate & wires[i] | fc_gate
        l= len(wires)
        for i in range(l):
            cu3_gate = CU3Gate()
            cu3_gate = cu3_gate(G2_param[i][0],G2_param[i][1],G2_param[i][2])
            cu3_gate.cargs=[wires[(l-i)%l]]
            cu3_gate.targs=[wires[l-i-1]]
            cu3_gate & [wires[(l-i)%l],wires[l-i-1]]|fc_gate  
            #cu3_gate |fc_gate  ([(l-i)%len(G2_param),l-i-1])
        for gate in fc_gate.gates:
            gate._requires_grad = True
        return fc_gate
class QCNNLayer(QNNLayer):
    def __init__(self, data_qubits, result_qubit, device,):
        super().__init__(data_qubits, result_qubit, device)
    def __call__(self,  params,qcnov:qconv2,pool:pool,fc:FC):
        """
        Args:
            params(tuple): the parm of gates,it is expected to be a 1*4 tuple
            tuple[0] contains params for qconv
            tuple[1] contains params for pool
            tuple[2] contains params for fc's G_param:list
            tuple[3] contains params for fc's G2_param:list
        """
    def circuit_layer(self,  params,qconv:qconv2,pool:pool,wires:list,com_gate:CompositeGate):
        """
        Args:
            params(tuple): the parm of gates,it is expected to be a 1*4 tuple
            tuple[0] contains params for qconv
            tuple[1] contains params for pool
            tuple[2] contains params for fc's G_param:list
            tuple[3] contains params for fc's G2_param:list
        """
        index_param0 = 0
        index_param1= 0
        new_wires=list()
        n_wires=len(wires)
        for i in range(1,n_wires,2):  # one layer contains two conv and one pool
            wire_list1=[wires[i],wires[(i+1)%n_wires]]
            wire_list2=[wires[i-1],wires[(i)%n_wires]]
            qconv("0",wires=wire_list1,param= params[0][index_param0]) |com_gate
            index_param0=index_param0+1
            qconv("0",wires=wire_list2,param= params[0][index_param0]) |com_gate
            index_param0=index_param0+1
            pool("0",param=params[1][index_param1],wires=wire_list2)|com_gate
            index_param1=index_param1+1
        
        for i in range(0,len(wires),2):
            new_wires.append(i)
        wires=new_wires
        
        return com_gate,wires
