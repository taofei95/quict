<<<<<<< HEAD
from tensorflow.keras.layers import Layer
import tensorflow as tf
from  QuICT.core.circuit import Circuit
from QuICT.simulation.simulator import Simulator
from QuICT.algorithm.quantum_machine_learning.utils.hamiltonian import Hamiltonian
import numpy as np
import sympy
=======
# from tensorflow.keras.layers import Layer
import tensorflow as tf
from QuICT.core.circuit import Circuit
from QuICT.simulation.simulator import Simulator
from QuICT.algorithm.quantum_machine_learning.utils.hamiltonian import Hamiltonian
import numpy as np
from QuICT.algorithm.quantum_machine_learning.differentiator.adjoint import AdjointDifferentiator
import sympy
from QuICT.algorithm.quantum_machine_learning.utils.expectation import Expectation
from QuICT.core.gate import *
import time
>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd
class PQC(tf.keras.layers.Layer):
    """
    Parametrized Quantum Circuit (PQC) Layer.
    """

    def __init__(
            self,
            model_circuit,
<<<<<<< HEAD
            pargs,
=======
            model_pargs,
>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd
            sim:Simulator,
            operators:Hamiltonian,
            differentiator=None,
            repetitions=None,
            *,
            backend='noiseless',
            
            initializer=tf.keras.initializers.RandomUniform(0, 2 * np.pi),
            regularizer=None,
            constraint=None,
<<<<<<< HEAD
            **kwargs,
    ):
        
        super().__init__(**kwargs)
        # Ingest model_circuit.
        self._model_circuit = model_circuit
=======
            opti = tf.keras.optimizers.Adam(),
            **kwargs,
    ):

        super().__init__(**kwargs)
        # Ingest model_circuit.
        self._model_circuit = model_circuit
        self._model_pargs = model_pargs
>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd
        self._sim = sim
        # Ingest operators.
        self._operators = operators
        # Ingest and promote repetitions.
        # Set backend and differentiator.
<<<<<<< HEAD
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

=======
        self._differ = differentiator
        self._executor = Expectation(self._model_circuit,self._operators,differentiator)
        self._opti = opti
        # Set additional parameter controls.
>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd

    def build(self, input_shape):
        """Keras build function."""
        super().build(input_shape)

    def call(self, inputs):
<<<<<<< HEAD
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
=======
        """Keras call function."""
        sv = self._sim.run(self._model_circuit)
        # sv= sv['data']['state_vector']
        if isinstance(self._differ, Adjoint):
            X.targs = [0]
            self._differ.run(
                self._model_circuit,
                variables=self._model_pargs,
                state_vector=sv,
                expectation_op=self._operators,
            )
        return

    def backward(self,):
        return

    def train(self,):
        sv = self._sim.run(self._model_circuit)
        self._differ.run(
            self._model_circuit,
            variables=self._model_pargs,
            state_vector=sv,
            expectation_op=self._operators,
        )
        # gradient update
        
        grads = []
        vars = []

        start_time = time.time()
        for item in self._model_pargs:  # Variable to tf.Variable
            grads.append(item.grads)
            vars.append(tf.Variable(item.pargs))
        middle_time = time.time() - start_time 
        print(str(middle_time))
        middle_time = time.time()
        self._opti.apply_gradients(zip(grads, vars))
        update_time = time.time() - middle_time
        print(update_time)

        for idx in range(len(vars)):  # tf.Variable to Variable
            self._model_pargs.pargs[idx] = vars[idx].numpy()
        self._model_circuit.update(
            self._model_pargs
        )  # Variable update to circuit.gates.pargs

        return self._model_circuit


@tf.function
def custom_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true)
    y_pred = tf.map_fn(lambda x: 1.0 if x >= 0 else -1.0, y_pred)
    return tf.keras.backend.mean(tf.keras.backend.equal(y_true, y_pred))


def custom_print(params):  # tool method used to debug
    see = []
    if isinstance(params, Circuit):
        for gate in params.gates:
            if gate.variables == 0:
                continue
            for i in range(len(gate.pargs)):
                see.append(gate.pargs[i].pargs)
    elif isinstance(params, Variable):
        see.append(params.pargs)
    else:
        for i in range(3):
            see.append(params[i].pargs)

    print(see)
    print("-------------")
    return


from QuICT.simulation.state_vector import StateVectorSimulator
if __name__ == '__main__':  
    opti = tf.keras.optimizers.Adam()
    params = Variable(np.random.random(200))
    sim = StateVectorSimulator(device="GPU")
    wideth = 5
    cir = Circuit(wideth)
    H | cir
    for i in range(wideth):
        Rxx(params[i]) | cir([i, (i + 1) % wideth])

    differ = Adjoint(device="GPU")

    variables = Variable(np.array([1.8, -0.7, 2.3]))
    circuit = Circuit(3)
    H | circuit
    Rx(0.2) | circuit(1)
    Rzx(variables[1] * 3 + 0.6) | circuit([0, 1])
    Rzz(variables[1]) | circuit([0, 1])
    Rzx(variables[2]) | circuit([0, 1])
    Rzx(3) | circuit([0, 1])
    cir = circuit

    # differ.run(circuit, sv, X)
    ham = Hamiltonian([[1, "Y1"]])
    pqc = PQC(model_circuit=cir,differentiator=differ,model_pargs= variables,sim=sim,operators = ham,opti=opti)
    for epoch in range(30):
        cir_trained = pqc.train()
        custom_print(cir_trained)

    # excitation_input = tf.keras.Input(shape=(), )
    # quantum_model = pqc(excitation_input)
    # qcnn_model = tf.keras.Model(inputs=[excitation_input], outputs=[quantum_model])
    # qcnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.02),
    #                loss=tf.losses.mse,
    #                )

>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd
