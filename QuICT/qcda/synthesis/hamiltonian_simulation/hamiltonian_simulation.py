from QuICT.qcda.synthesis.hamiltonian_simulation.TS_method import *

from QuICT.qcda.synthesis.hamiltonian_simulation.Trotter import *

from QuICT.qcda.synthesis.hamiltonian_simulation.unitary_matrix_encoding \
    import *

from QuICT.core.gate import *


class HamiltonianSimulation():

      def __init__(self, method):
        self.method = method
        assert self.method == "TS" or self.method ==  "Trotter" or self.method ==  "QSP", "Please select 'Trotter', 'TS' or 'QSP' method."
    def execute(self, hamiltonian, time, initial_state, error = 0.1, max_order = 20):
        coefficient_array, hamiltonian_array = hamiltonian
        assert len(initial_state) == len(hamiltonian_array[0][0]), "The initial state size must equal to hamiltonian row number."
        if self.method == "TS":
            TS_method_circuit, TS_method_gate_width, steps = TS_method(coefficient_array, hamiltonian_array, time, error, max_order, initial_state)
            return TS_method_circuit, steps
        elif self.method == "Trotter":

            trotter_method_gate = Trotter(
                hamiltonian, time, error).output_gate()

            initial_state_gate | cg(
                [i for i in range(initial_state_gate_width)])

            trotter_method_gate | cg(
                [i for i in range(initial_state_gate_width)])

            return cg
