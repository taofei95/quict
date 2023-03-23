import numpy as np
import torch 
from QuICT.core import Circuit
from QuICT.core.gate.composite_gate import CompositeGate
from QuICT.core.gate import BasicGate
from QuICT.simulation.simulator import Simulator
from QuICT.algorithm.quantum_machine_learning.utils.ansatz import Ansatz
from QuICT.algorithm.quantum_machine_learning.utils.hamiltonian import Hamiltonian
from typing import Union
class Adjoint: 
    def __init__(self,cir: Union[Circuit,Ansatz],sim :Simulator) -> None:
        """
        Args:
        cir(Circuit): the circuit which is needed to calculate gradient
        """
        self._cir =cir
        self._sim =sim

    def get_grad(self,ham:Hamiltonian):
        sim = Simulator(device='GPU')
        _,state_vector = sim.forward(self._cir,ham=ham)
        vec_k= state_vector.copy() 
        n_qubits = int(np.log2(len(state_vector)))
        d_k=2*np.dot(ham.get_hamiton_matrix(n_qubits).T.conj() ,vec_k )
        grad = []
        real_sim = sim._load_simulator()
        real_sim.initial_circuit(self._cir)
        real_sim._vector = state_vector
        for gate in reversed(self._cir.gates):
            targ=gate.targets
            if isinstance(targ,int):
                targ=[targ,]
            temp_cir = Circuit(n_qubits)

            inver_gate = gate.inverse()
            inver_gate | temp_cir

            state_vector = sim.run(temp_cir,state_vector=vec_k)
            state_vector = state_vector['data']['state_vector']
            
            d_U_k =np.dot(d_k,state_vector.conj())
            d_k=np.dot(temp_cir.matrix,d_k)
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
        
    def get_param_shift(self,idx_gate:int,idx_param:int,shift_type:int):
        """
        Args:
        shift_type(int): 0 for left shift and 1 for right shift
        """
        return