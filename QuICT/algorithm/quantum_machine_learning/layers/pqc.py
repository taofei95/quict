import numpy as np
import tensorflow as tf
from QuICT.core.gate import *
from QuICT.core.circuit import Circuit
from QuICT.simulation.state_vector import StateVectorSimulator
from QuIC.algorithm.quantum_machine_learning.differentiator.adjoint.adjoint_differentiator import AdjointDifferentiator
from QuICT.algorithm.quantum_machine_learning.utils.hamiltonian import Hamiltonian
from QuICT.algorithm.quantum_machine_learning.differentiator import *
from QuICT.algorithm.quantum_machine_learning.utils.expectation import Expectation
from typing import List
#  tf.tile(self._model_circuit, [circuit_batch_dim])
def tile(inputs,multiples):
    outputs = []
    def custom_copy(inner_inputs):
        if isinstance(inner_inputs,Circuit):
            qbit = inner_inputs.width
            inner_output = Circuit(qbit)
            for gates in inner_inputs.flatten_gates():
                gates|inner_output
            return inner_output
        elif isinstance(inner_inputs,Hamiltonian):
            pauli_str = inner_inputs.pauli_str
            hami = Hamiltonian(pauli_str)
            return hami
        elif isinstance(inner_inputs,AdjointDifferentiator):  # it's may be not correct here
            return inner_inputs.copy()

        return inner_inputs.copy()
    if len(multiples) == 1:
        for i in range(multiples[0]):
            outputs.append(custom_copy(inputs))
    else:
        raise ValueError('tile method now do not support higher dim than 1-D')
    
    return outputs
class PQC(tf.keras.layers.Layer):
    __PRECISION = ["single", "double"]

    def __init__(
        self,
        model_circuit: Circuit,
        params: Variable,
        hamiltonian,
        precision: str = "double",
        *,
        repetitions=None,
        differentiator=None,
        initializer=tf.keras.initializers.RandomUniform(0, 2 * np.pi),
        regularizer=None,
        constraint=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._model_circuit = model_circuit
        self._hamiltonian = hamiltonian
        self._differ = differentiator
        self._paramter = params

    def append_layer(self,circuit_list,append_circuit_list):
        def add_circuit(circuit,append_circuit):
            if isinstance(circuit,Circuit):
                circuit.extend(append_circuit)
            return circuit
        if len(circuit_list) != len(append_circuit_list):
            raise ValueError('inputs should have the same size')
        output_layer = []
        for i in range(len(circuit_list)):
            output_layer.append(add_circuit(circuit_list[i],append_circuit_list[i]))
        return output_layer

        
    def build(self, input_shape):
        """Keras build function."""
        super().build(input_shape)

    

    def call(self, inputs:List[Circuit]):
        circuit_batch_dim = len(inputs)
        tiled_up_model = tile(self._model_circuit, [circuit_batch_dim])
        model_appended = self.append_layer(inputs, append=tiled_up_model)  
        tiled_up_parameters = tile(self.parameters, [circuit_batch_dim])
        tiled_up_operators = tile(self._hamiltonian, [circuit_batch_dim])
        tiled_up_differ = tile(self._differ,[circuit_batch_dim])
        self._executor = Expectation()
    
        return self._executor(model_appended, 
                              tiled_up_parameters,
                              hamiltonian=tiled_up_operators,
                              differentiator= tiled_up_differ)
    
