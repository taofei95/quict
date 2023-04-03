from tensorflow.keras.layers import Layer
import tensorflow as tf
from  QuICT.core.circuit import Circuit
from QuICT.simulation.simulator import Simulator
from QuICT.algorithm.quantum_machine_learning.utils.hamiltonian import Hamiltonian
import numpy as np
import sympy
class PQC(tf.keras.layers.Layer):
    """
    Parametrized Quantum Circuit (PQC) Layer.
    """

    def __init__(
            self,
            model_circuit,
            pargs,
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
        self.initializer = tf.keras.initializers.get(initializer)
        self.regularizer = tf.keras.regularizers.get(regularizer)
        self.constraint = tf.keras.constraints.get(constraint)

        # Weight creation is not placed in a Build function because the number
        # of weights is independent of the input shape.
        
        self.parameters = self.add_weight('parameters',
                                          shape=self._symbols.shape,
                                          initializer=self.initializer,
                                          regularizer=self.regularizer,
                                          constraint=self.constraint,
                                          dtype=tf.float32,
                                          trainable=True)
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


    def build(self, input_shape):
        """Keras build function."""
        super().build(input_shape)

    def call(self, inputs):
        """Keras call function.""" 
        return self._sim.forward(self._model_circuit,ham=self._operators)
    def backward(self,):
        return
            
        

from QuICT.core.gate import *
if __name__ == '__main__':  
    wideth = 5
    cir = Circuit(wideth)
    for i in range(wideth):
        rx = Rx(0.1) 
        rx.set_requires_grad(True)
        rx | cir
    params = np.random.random(200)
    sim = Simulator()
    ham = Hamiltonian([[0.4, 'Y0', 'X1', 'Z2', 'I5'], [0.6]])
    pqc = PQC(cir,params,sim,ham)
    pqc.call(Circuit(wideth))