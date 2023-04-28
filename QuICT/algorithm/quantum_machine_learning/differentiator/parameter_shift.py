import numpy as np
from QuICT.core.gate import *
from QuICT.core import Circuit
from QuICT.algorithm.quantum_machine_learning.utils import Hamiltonian
from QuICT.simulation.state_vector.statevector_simulator import StateVectorSimulator
from QuICT.algorithm.quantum_machine_learning.differentiator.adjoint.adjoint_differentiator import AdjointDIfferentiator
from QuICT.algorithm.quantum_machine_learning.utils.expectation import Expectation
def get_expe(sv,hamiltonian,n_qubit):
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
        # e_val = np.dot(e_val, ket).real
        return e_val

class ParameterShift(AdjointDifferentiator): 
    def __init__(self) -> None:
        super().__init__()
        """
        Args:
        cir(Circuit): the circuit which is needed to calculate gradient
        """
        self._gate_calculator = StateVectorSimulator(
            self._device, self._precision, self._device_id, self._sync
        )
    def run(
        self,
        circuit: Circuit,
        variables: Variable,
        state_vector: np.ndarray,
        expectation_op: Hamiltonian,
    ):
        self._circuit = circuit
        self._variables = variables
        self._expectation_op =  expectation_op

        for gate in self._circuit.fast_gates:
            if gate.variables == 0:
                continue
            for  i in range(gate.variables):
                e_shift = np.array( [0.0 ,0.0]  )
                for j in range(2):
                    e_shift[j]=self.get_param_shift(gate,i,j)
                grad = 1.0*(e_shift[1]-e_shift[0])

                gate.pargs[i].grads = (
                    grad
                    if abs(gate.pargs[i].grads) < 1e12
                    else grad*gate.pargs[i].grads
                )
                index_global = gate.pargs[i].index
                self._variables.grads[index_global] += gate.pargs[i].grads

    
    def get_param_shift(self,gate,idx_param:int,shift_type:int):
        """
        Args:
        shift_type(int): 0 for left shift and 1 for right shift
        """
        if shift_type !=0 and shift_type != 1:
            raise  ValueError("shift_type must be 0 or 1")
        i =idx_param
        sim = self._sim
        shift = 0.5
        shift_type = 1.0 if shift_type==1 else -1.0

        old_param = gate.pargs[i].copy()
        gate.pargs[i]= gate.pargs[i]+shift*shift_type
        sv = sim.run(self._circuit)
        e_val= get_expe(sv,hamiltonian=self._expectation_op,n_qubit=self._circuit.width())
        gate.pargs[i] = old_param

        return e_val


    