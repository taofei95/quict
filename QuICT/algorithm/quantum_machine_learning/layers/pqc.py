import numpy as np
import tensorflow as tf


from QuICT.core.gate import *
from QuICT.core.circuit import Circuit
from QuICT.simulation.state_vector import StateVectorSimulator
from QuICT.algorithm.quantum_machine_learning.utils.hamiltonian import Hamiltonian
from QuICT.algorithm.quantum_machine_learning.differentiator import *


class PQC(tf.keras.layers.Layer):
    def __init__(
        self,
        model_circuit,
        hamiltonian,
        *,
        repetitions=None,
        differentiator=None,
        initializer=tf.keras.initializers.RandomUniform(0, 2 * np.pi),
        regularizer=None,
        constraint=None,
        **kwargs,
    ):  
        super().__init__(**kwargs)

    def build(self, input_shape):
        """Keras build function."""
        super().build(input_shape)

    def call(self, inputs):
        """Keras call function."""
        return
