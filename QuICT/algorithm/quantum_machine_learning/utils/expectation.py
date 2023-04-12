# from tensorflow.keras.layers import Layer
import tensorflow as tf
from QuICT.core.circuit import Circuit
from QuICT.simulation.state_vector import StateVectorSimulator
from QuICT.algorithm.quantum_machine_learning.utils.hamiltonian import Hamiltonian
import numpy as np
from QuICT.algorithm.quantum_machine_learning.differentiator.adjoint import Adjoint
import sympy


class Expectation(tf.keras.layers.Layer):
    def __init__(
        self,
        model_circuit,
        operators,
        repetitions=None,
        backend="noiseless",
        differentiator=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._model_circuit = model_circuit
        self._differ = differentiator
        self._operators = operators
        sim = StateVectorSimulator()
        self.sv = sim.run(self._model_circuit)

    def build(self, input_shape):
        """Keras build function."""
        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        bra = self.sv.copy().conj()
        ket = self.sv.copy()
        e_val = np.dot(
            np.dot(
                bra,
                self._operators.get_hamiton_matrix(
                    n_qubits=self._model_circuit.width()
                ),
            ),
            ket,
        ).real
        # e_val = np.dot(e_val, ket).real
        return np.dot(
            np.dot(
                bra,
                self._operators.get_hamiton_matrix(
                    n_qubits=self._model_circuit.width()
                ),
            ),
            ket,
        ).real

