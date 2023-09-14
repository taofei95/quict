from .ts_method import truncate_series
from .trotter import *


class HamiltonianSimulation():
    def __init__(self, method):
        """
        Args:
             method: string, either Ts or Trotter
        """
        self.method = method
        assert self.method == "TS" or self.method == "Trotter", "Please select 'Trotter'or 'TS' method."

    def execute(self, hamiltonian, time, initial_state, error=0.1, max_order=20):
        """
        Args:
            hamiltonian: string, either Ts or Trotter
            time: string, either Ts or Trotter
            initial_state: string, either Ts or Trotter
            error: string, either Ts or Trotter
            max_order: string, either Ts or Trotter
        Returns:
            if TS method:
            circuit: QuICT circuit simulate e^-iHt/r
            c_width: circuit with
            time_steps: Time steps to give error bounded by 0.1 in each time step iteration
            order: order to truncated taylor series
            summed_coefficient: summed coefficient of equation 6(See TS method)
            expected_error: bound error
            amplification_size: expect 2, but in most of case it is a float number close to 2.
            approximate_time_evolution_operator: the approximated e^-iHt/r

            Noting that, to get e^-iHt, you take measurements on ancilla qubit, get the final state,
            Then rerun the circuit with initial state prepares with the last final state.

            If Trotter method:
            circuit: a circuit calculate e^-iHt
        """
        if self.method == "TS":
            coefficient_array, hamiltonian_array = hamiltonian
            assert len(initial_state) == len(hamiltonian_array[0][0]), ("The initial state size must equal to hamiltonian"
                                                                        " row number.")
            circuit, circuit_info_dictionary = truncate_series(coefficient_array, hamiltonian_array,
                                                               time,
                                                               error,
                                                               max_order,
                                                               initial_state)
            return circuit, circuit_info_dictionary
        elif self.method == "Trotter":
            circuit, circuit_info_dictionary = trotter(hamiltonian, time, error, initial_state)
            return circuit, circuit_info_dictionary