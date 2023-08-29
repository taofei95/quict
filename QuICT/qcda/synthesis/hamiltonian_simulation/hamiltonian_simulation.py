from .TS_method import *
from .Trotter import *
from QuICT.core.gate import *
class HamiltonianSimulation():
    def __init__(self, method):
        self.method = method
        assert self.method != ("TS" or "Trotter" or "QSP"), "Please select 'Trotter', 'TS' or 'QSP' method."
    def execute(self, hamiltonian, time, initial_state, error = 0.01, max_order = 20):
        initial_state_gate, _ = gates_B(initial_state)
        initial_state_gate_width = initial_state_gate.width()
        cg = CompositeGate()
        if self.method == "TS":
            coefficient_array, hamiltonian_array = hamiltonian
            TS_method_gate = TS_method(coefficient_array, hamiltonian_array, time, error, max_order)
            TS_method_gate_width = TS_method_gate.width()
            TS_method_gate | cg([i for i in range(TS_method_gate_width)])
            initial_state_gate | cg([TS_method_gate_width+i for i in range(initial_state_gate_width)])
            return cg
        elif self.method =="Trotter":
            trotter_method_init = Trotter(hamiltonian, time, error, initial_state)
            trotter_method_gate = trotter_method_init.output_gate()
            initial_state_gate | cg([i for i in range(initial_state_gate_width)])
            trotter_method_gate | cg([i for i in range(initial_state_gate_width)])
            return cg