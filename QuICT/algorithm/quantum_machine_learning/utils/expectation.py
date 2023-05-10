# from tensorflow.keras.layers import Layer
import tensorflow as tf
from QuICT.core.circuit import Circuit
from QuICT.simulation.state_vector import StateVectorSimulator
from QuICT.algorithm.quantum_machine_learning.utils.hamiltonian import Hamiltonian
import numpy as np
from QuICT.algorithm.quantum_machine_learning.differentiator.adjoint import AdjointDifferentiator
import sympy


class Expectation(tf.keras.layers.Layer):
    def __init__(
        self,
        backend="noiseless",
        **kwargs,
    ):
        super().__init__(**kwargs)
   
    
    def get_expe(self,sv,hamiltonian,n_qubit):
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
    def build(self, input_shape):
        """Keras build function."""
        super().build(input_shape)

    def call(self, 
        model_circuit,
        hamiltonian,
        params,
        differentiatorr=None,
        repetitions=None,*args, **kwargs):
        foward_pass_vals = []
        tile_up_sv = []
        for i in range(len(model_circuit)):
            sim =  StateVectorSimulator()
            sv = sim.run(model_circuit[i])
            tile_up_sv.append(sv)
            n_qubit = model_circuit[i].width()
            expectation_value = self.get_expe(sv,hamiltonian[i],n_qubit)
            foward_pass_vals.append(expectation_value)
        
        # TODO differtiaor should be called in this method, so follows are not finished code
        def todo_grad(model_circuit,hamiltonian,params,differentiator,tile_up_sv): 
            
            for i in range(len(model_circuit)):
                 differentiator[i].run(model_circuit[i],params[i],tile_up_sv[i],hamiltonian[i])
            return
        return foward_pass_vals,todo_grad

