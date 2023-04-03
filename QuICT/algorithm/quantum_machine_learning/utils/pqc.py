#from tensorflow.keras.layers import Layer
import tensorflow as tf
from QuICT.core.circuit import Circuit
from QuICT.simulation.simulator import Simulator
from QuICT.algorithm.quantum_machine_learning.utils.hamiltonian import Hamiltonian
import numpy as np
from QuICT.algorithm.quantum_machine_learning.differentiator.adjoint import Adjoint
import sympy
class PQC(tf.keras.layers.Layer):
    """
    Parametrized Quantum Circuit (PQC) Layer.
    """

    def __init__(
            self,
            model_circuit,
            model_pargs,
            sim:Simulator,
            operators:Hamiltonian,
            differentiator=None,
            repetitions=None,
            *,
            backend='noiseless',
            
            initializer=tf.keras.initializers.RandomUniform(0, 2 * np.pi),
            regularizer=None,
            constraint=None,
            **kwargs,
    ):
        
        super().__init__(**kwargs)
        # Ingest model_circuit.
        self._model_circuit = model_circuit
        self._sim = sim
        # Ingest operators.
        self._operators = operators
        # Ingest and promote repetitions.
        # Set backend and differentiator.
        self._executor = differentiator
        # Set additional parameter controls.
        '''
        pargs_index = 0
        for g_id, gate in enumerate(self._model_circuit.gates):
            if gate.is_requires_grad:
                pargs_copy = gate.pargs
                if isinstance(pargs_copy,int):
                    pargs_copy = [pargs_copy]
                for i in range(len(pargs_copy)):
                    pargs_copy[i] = pargs[pargs_index]
                    pargs_index += 1
                gate.pargs = pargs_copy.copy()
                self._model_circuit.replace_gate(g_id,gate)
      
        '''
        
        


    def build(self, input_shape):
        """Keras build function."""
        super().build(input_shape)

    def call(self, inputs):
        """Keras call function.""" 
        sv = self._sim.run(self._model_circuit)
        sv= sv['data']['state_vector']
        if  isinstance(self._executor ,Adjoint):
            self._executor(self._model_circuit,sv)

        grad_list_see = []
        for gate in self._model_circuit.gates:
            if gate.variables > 0:
                for parg in gate.pargs:
                    grad_list_see.append(parg.grad)
        print(grad_list_see)
        return 
    def backward(self,):
        return
            
        

from QuICT.core.gate import *
if __name__ == '__main__':  
    wideth = 5
    cir = Circuit(wideth)

    params = np.random.random(200)
    sim = Simulator(device='GPU')
    ham = Hamiltonian([[0.4, 'Y0', 'X1', 'Z2', 'I5'], [0.6]])
    ad = Adjoint()
    pqc = PQC(model_circuit=cir,differentiator=ad,model_pargs= params,sim=sim,operators = ham)
    pqc.call(1)