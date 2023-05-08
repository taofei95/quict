import numpy as np
import tensorflow as tf


from QuICT.core.gate import *
from QuICT.core.circuit import Circuit
from QuICT.simulation.state_vector import StateVectorSimulator
from QuICT.algorithm.quantum_machine_learning.utils.hamiltonian import Hamiltonian
from QuICT.algorithm.quantum_machine_learning.differentiator import *


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
        if model_circuit.count_training_gate() == 0:
            raise ValueError

        self._params = params
        self._precision = tf.float64 if precision == "double" else tf.float32

        # Set additional parameter controls.
        self.initializer = tf.keras.initializers.get(initializer)
        self.regularizer = tf.keras.regularizers.get(regularizer)
        self.constraint = tf.keras.constraints.get(constraint)

        # Weight creation is not placed in a Build function because the number
        # of weights is independent of the input shape.
        self.parameters = self.add_weight(
            "parameters",
            shape=self._params.shape,
            initializer=self.initializer,
            regularizer=self.regularizer,
            constraint=self.constraint,
            dtype=self._precision,
            trainable=True,
        )

    def build(self, input_shape):
        """Keras build function."""
        super().build(input_shape)

    def call(self, inputs):
        """Keras call function."""
        circuit_batch_dim = tf.gather(tf.shape(inputs), 0)
        tiled_up_model = tf.tile(self._model_circuit, [circuit_batch_dim])
        model_appended = self._append_layer(inputs, append=tiled_up_model)
        tiled_up_parameters = tf.tile([self.parameters], [circuit_batch_dim, 1])
        tiled_up_operators = tf.tile(self._operators, [circuit_batch_dim, 1])

        # this is disabled to make autograph compilation easier.
        # pylint: disable=no-else-return
        if self._analytic:
            return self._executor(
                model_appended,
                symbol_names=self._symbols,
                symbol_values=tiled_up_parameters,
                operators=tiled_up_operators,
            )
        else:
            tiled_up_repetitions = tf.tile(self._repetitions, [circuit_batch_dim, 1])
            return self._executor(
                model_appended,
                symbol_names=self._symbols,
                symbol_values=tiled_up_parameters,
                operators=tiled_up_operators,
                repetitions=tiled_up_repetitions,
            )
        # pylint: enable=no-else-return
