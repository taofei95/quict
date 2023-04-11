#from tensorflow.keras.layers import Layer
import tensorflow as tf
from QuICT.core.circuit import Circuit
from QuICT.simulation.simulator import Simulator
from QuICT.algorithm.quantum_machine_learning.utils.hamiltonian import Hamiltonian
import numpy as np
from QuICT.algorithm.quantum_machine_learning.differentiator.adjoint import Adjoint
import sympy
from QuICT.algorithm.quantum_machine_learning.utils.expectation import Expectation
from QuICT.core.gate import *
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
        self._differ = differentiator
        self._executor = Expectation(self._model_circuit,self._operators,differentiator)
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

    def get_param_circuit(self,):
        param_list = []
        for gate in self._model_circuit.gates:
            if gate.variables>0:
                for param in gate.pargs:
                    param_list.append(param)
        return param_list
                    
    def call(self, inputs):
        """Keras call function.""" 
        sv = self._sim.run(self._model_circuit)
        #sv= sv['data']['state_vector']
        if  isinstance(self._differ ,Adjoint):
            X.targs = [0]
            self._differ.run(self._model_circuit, sv, X)
            opti = tf.keras.optimizer.experimental.SGD(learning_rage=0.1)
            see  = self.get_param_circuit()
            opti.apply_gradients(zip(grads,))

        grad_list_see = []
        for gate in self._model_circuit.gates:
            if gate.variables > 0:
                for parg in gate.pargs:
                    grad_list_see.append(parg.grads)
        #print(grad_list_see)
        #return self._executor(None)
    def backward(self,):
        return
            
@tf.function
def custom_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true)
    y_pred = tf.map_fn(lambda x: 1.0 if x >= 0 else -1.0, y_pred)
    return tf.keras.backend.mean(tf.keras.backend.equal(y_true, y_pred))     

from QuICT.simulation.state_vector import StateVectorSimulator
if __name__ == '__main__':  

    params = Variable( np.random.random(200))
    sim = StateVectorSimulator(device="GPU")
    wideth = 5
    cir = Circuit(wideth)
    H | cir
    for i in range(wideth):
        Rxx(params[i]) | cir([i, (i+1)%wideth])

    differ = Adjoint(device="GPU")
    #differ.run(circuit, sv, X)
    ham = Hamiltonian([[0.4, 'Y0', 'X1', 'Z2', 'I5'], [0.6]])
    pqc = PQC(model_circuit=cir,differentiator=differ,model_pargs= params,sim=sim,operators = ham)
    
    excitation_input = tf.keras.Input(shape=(), )
    quantum_model = pqc(excitation_input)
    qcnn_model = tf.keras.Model(inputs=[excitation_input], outputs=[quantum_model])
    qcnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.02),
                   loss=tf.losses.mse,
                   metrics=[custom_accuracy])
    

