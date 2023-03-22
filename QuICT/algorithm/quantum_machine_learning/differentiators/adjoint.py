import numpy as np
import torch 
from QuICT.core import Circuit
from QuICT.core.gate.composite_gate import CompositeGate
from QuICT.core.gate import BasicGate
from QuICT.simulation.simulator import Simulator
from QuICT.algorithm.quantum_machine_learning.utils.ansatz import Ansatz
from typing import Union
class Adjoint: 
    def __init__(self,cir: Union[Circuit,Ansatz],sim :Simulator) -> None:
        """
        Args:
        cir(Circuit): the circuit which is needed to calculate gradient
        """
        self._cir =cir
        self._sim =sim
    def get_grad_0(self,):
        sim = Simulator()
        _,state_vector = sim.forward(self._cir)
        vec_k= [state_vector.copy(),state_vector.copy()]  # vec_k[0]:vector fai_k,vec_k[1]: vector fai_k+1
        d_k = [[],np.ones(2)]  #  d_k[0]:partial adjont derivative of  small_k respects to fai_k
        grad = []
        for gate in reversed(self._cir.gates):
            targ=gate.targets
            if isinstance(targ,int):
                targ=[targ,]
            small_k= [[],np.zeros(2*len(targ),dtype=complex),] # activate part of vector_k
            j=0
            for i in targ:
                small_k[1][j] = vec_k[1][2*i]
                small_k[1][j+1] = vec_k[1][2*i+1]
                j=j+2
            inver_gate = gate.inverse()
            if len(inver_gate.matrix)!=2:
                matrix = inver_gate.matrix[2:,2:]
            else:
                matrix = inver_gate.matrix
            small_k[0] =np.dot(matrix,small_k[1])
            j=0
            for i in targ:
                vec_k[0][2*i] = small_k[0][j]
                vec_k[0][2*i+1] = small_k[0][j+1]  # now we have vector fai_k
                j=j+2
            d_k[0]=np.dot(matrix,small_k[0])
            d_U_k =np.dot(d_k[1],small_k[0].T.conj())

            vec_k[1]=vec_k[0]
            d_k[1]=d_k[0]
            small_k[1]=small_k[0]

                # now we'll get the grad of paramter from d_U_k
            if not gate.is_requires_grad():
                continue
            if isinstance(gate.pargs,int):
                param_list = [gate.pargs]
            else:
                param_list = gate.pargs
            for i in range(len(param_list)):
                grad.append(np.sum(np.dot(d_U_k,gate.parti_deri_adj[i])))  
                # note : only Ry，Rz，U3，CU3 gate have method parti_deri_adj now

        return grad
    def get_grad_1(self,):
        #if self._cir.device.type == "cpu" or self._cir.device.type == "cuda:0":
        if True:
            state_vector = self._cir.forward()
        else:  # Adjoint do not support GPU 
            raise ' Adjoint do not support GPU now '
        vec_k= [state_vector,state_vector]  # vec_k[0]:vector fai_k,vec_k[1]: vector fai_k+1
        d_k = [[],np.ones(2)]  #  d_k[0]:partial adjont derivative of  small_k respects to fai_k
        grad = []
        for gate in reversed(self._cir.gates):
            targ=gate.targets
            if isinstance(targ,int):
                targ=[targ,]
            small_k= torch.tensor([[],torch.tensor.zeros(2*len(targ),dtype=complex),]) # activate part of vector_k
            j=0
            for i in targ:
                small_k[1][j] = vec_k[1][2*i]
                small_k[1][j+1] = vec_k[1][2*i+1]
                j=j+2
            inver_gate = gate.inverse()
            if len(inver_gate.matrix)!=2:
                matrix = inver_gate.matrix[2:,2:]
            else:
                matrix = inver_gate.matrix
            small_k[0] =np.dot(matrix,small_k[1])
            j=0
            for i in targ:
                vec_k[0][2*i] = small_k[0][j]
                vec_k[0][2*i+1] = small_k[0][j+1]  # now we have vector fai_k
                j=j+2
            d_k[0]=np.dot(matrix,small_k[0])
            d_U_k =np.dot(d_k[1],small_k[0].T.conj())

            vec_k[1]=vec_k[0]
            d_k[1]=d_k[0]
            small_k[1]=small_k[0]

                # now we'll get the grad of paramter from d_U_k
            if not gate.pargs.requires_grad:
                continue
            if isinstance(gate.pargs,int):
                param_list = [gate.pargs]
            else:
                param_list = gate.pargs
            for i in range(len(param_list)):
                grad.append(np.sum(np.dot(d_U_k,gate.parti_deri_adj[i])))  
                # note : only Ry，Rz，U3，CU3 gate have method parti_deri_adj now
        return grad
    def get_grad(self,):
        if isinstance(self._cir,Circuit):
            grad = self.get_grad_0()  # gate.matrix is np.array
        else:
            grad = self.get_grad_1()  # gate.matrix is tensor
        return grad
        
    def get_param_shift(self,idx_gate:int,idx_param:int,shift_type:int):
        """
        Args:
        shift_type(int): 0 for left shift and 1 for right shift
        """
        return