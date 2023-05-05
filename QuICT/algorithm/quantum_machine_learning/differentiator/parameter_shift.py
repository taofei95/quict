import numpy as np
import cupy as cp
from QuICT.core.gate import *
from QuICT.core import Circuit
from QuICT.algorithm.quantum_machine_learning.utils import Hamiltonian
from QuICT.simulation.state_vector.statevector_simulator import StateVectorSimulator
from QuICT.algorithm.quantum_machine_learning.differentiator.adjoint import (
    AdjointDifferentiator,
)
from QuICT.algorithm.quantum_machine_learning.utils.expectation import Expectation

def get_expe(sv,hamiltonian,n_qubit):
        if  not isinstance(sv,np.ndarray):
            sv = cp.asnumpy(sv)
        bra = sv.copy().conj()
        ket = sv.copy()
        e_val = np.dot(
            np.dot(
                bra,
                hamiltonian.get_hamiton_matrix(
                    n_qubits=n_qubit
                ),
            ),
            ket,
        ).real
    
        return e_val

class ParameterShift(AdjointDifferentiator): 
    def __init__(self,device,precision,gpu_device_id,sync) -> None:
        super().__init__(device,precision,gpu_device_id,sync)
        """
        Args:
        cir(Circuit): the circuit which is needed to calculate gradient
        """
        self._gate_calculator = StateVectorSimulator()
    def run(
        self,
        circuit: Circuit,
        variables: Variable,
        state_vector: np.ndarray,
        expectation_op: Hamiltonian,
    ):
        self._circuit = circuit
        self._expectation_op =  expectation_op
        for gate,_,_ in self._circuit.fast_gates:
            if gate.variables == 0:
                continue
            for  i in range(gate.variables):
                e_shift = np.array( [0.0 ,0.0]  )
                for j in range(2):
                    e_shift[j]=self.get_param_shift(gate,i,j)
                grad = 0.25*(e_shift[1]-e_shift[0])
                gate.pargs[i].grads= (
                    grad
                    if abs(gate.pargs[i].grads) < 1e-12
                    else grad*gate.pargs[i].grads
                )
                index_global = gate.pargs[i].index
                variables.grads[index_global] += gate.pargs[i].grads
        expectation = get_expe(state_vector,self._expectation_op,self._circuit.width())
        return variables, expectation

    
    def get_param_shift(self,gate,idx_param:int,shift_type:int):
        """
        Args:
        shift_type(int): 0 for left shift and 1 for right shift
        """
        if shift_type !=0 and shift_type != 1:
            raise  ValueError("shift_type must be 0 or 1")
        i =idx_param
        sim = StateVectorSimulator()
        shift = np.pi/2
        shift_type = 1.0 if shift_type==1 else -1.0

        old_param = gate.pargs[i].copy()
        gate.pargs[i].pargs= gate.pargs[i].pargs+shift*shift_type
        gate._is_matrix_update = True
        sv = sim.run(self._circuit)
        e_val= get_expe(sv,hamiltonian=self._expectation_op,n_qubit=self._circuit.width())
        gate.pargs[i]= old_param.copy()
        gate._is_matrix_update = True

        return e_val


    