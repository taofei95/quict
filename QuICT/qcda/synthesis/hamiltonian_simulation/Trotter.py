from QuICT.algorithm.quantum_machine_learning.utils.hamiltonian import *
from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.tools.exception.core import *
from QuICT.simulation.state_vector import StateVectorSimulator

import numpy as np
np.set_printoptions(suppress=True)


class Trotter:

    def __init__(self, pauli_string, t, eps, i_circuit=False, iterations=1):

        self.pauli_string = pauli_string
        self.h = Hamiltonian(pauli_string)
        self.n = max(max((self.h)._qubit_indexes)) + 1  # total qubits needed
        self.t = t  # Simulation time
        self.eps = eps  # error
        self.i_circuit = i_circuit  # initial state circuit
        self.iterations = iterations  # required iterations
        
    def gate(self):
        # construct the single step composite gate for H simu
        cgate = CompositeGate()
        ccgate = CompositeGate()

        gate_dict = {"X": X, "Y": Y, "Z": Z}

        for qbt_index, pauli_gate, ceoff in zip((self.h)._qubit_indexes, (self.h)._pauli_gates, (self.h)._coefficients):

            for gateindex, gate in zip(qbt_index, pauli_gate):  # left bracket gatew
                if gate_dict[gate] == X:
                    H | cgate(gateindex)
                elif gate_dict[gate] == Y:
                    U2(np.pi / 2, np.pi / 2) | cgate(gateindex)
                RZ = max(qbt_index)  # RZ = index of the central Rz
                CN = min(qbt_index)  # CN = index where CNOT gate starts

            for i in range(CN, RZ):  # left CNOT gates
                CX | cgate([i, i + 1])

            Rz(2 * (self.t / self.iterations) * ceoff) | cgate(RZ)  # central Rz gate

            for i in reversed(range(CN, RZ)):  # right CNOT gates
                CX | cgate([i, i + 1])

            for gateindex, gate in (zip(qbt_index, pauli_gate)):  # right bracket gate
                if gate_dict[gate] == X:
                    H | cgate(gateindex)

                elif gate_dict[gate] == Y:
                    U2(np.pi / 2, np.pi / 2) | cgate(gateindex)
        return cgate

    def Trotter_circuit(self):  # circuit with a single step cgate
        cir = Circuit(self.n)
        for i in range(self.iterations):
            delta_t = self.t / self.iterations
            gate = self.gate()
            gate | cir
        return cir

    def initialstate(self, random=False):  # Prepare the initial circuit and state (choose from random, 0s, given)
        if random is False:
            if self.i_circuit is False:
                icircuit = Circuit(self.n)
            else:
                icircuit = self.i_circuit
        elif random is True:
            icircuit = Circuit(self.n)
            icircuit.random_append(self.n)
        simu = StateVectorSimulator()
        result = simu.run(icircuit)
        return result, icircuit

    def accurate_finalstate(self):  # final state calculated by matrix mul.
        matrix = Hamiltonian.get_hamiton_matrix(self.h, self.n)
        Eval, Evec = np.linalg.eig(matrix)
        U = Evec
        U_inverse = np.linalg.inv(Evec)
        D = np.diag(Eval)
        exp_D = np.zeros((2**self.n, 2**self.n), dtype='complex_')
        for i in range(2**self.n):
            a = self.t * complex(0, -1) * (D[i][i])
            exp_D[i][i] = np.exp(a)
        mat_operator = np.matmul(np.matmul(U, exp_D), U_inverse)
        fstate = np.matmul(mat_operator, self.initialstate()[0])
        threshold = 1e-12
        mask = np.abs(fstate) < threshold
        fstate[mask] = 0
        return fstate

    def simulation_finalstate(self):  # final state found by circuit simulation
        simu = StateVectorSimulator()
        for i in range(self.iterations):
            crct = self.initialstate()[1]
            self.gate() | crct
        H_fstate = simu.run(crct)
        threshold = 1e-12
        mask = np.abs(H_fstate) < threshold
        H_fstate[mask] = 0
        return H_fstate
    
    
    def error(self):  # find theh absolute error between circuit
        distance = 0
        diff = self.accurate_finalstate() - self.simulation_finalstate()
        distance = distance + np.matmul(diff, np.conj(diff))

        magnitude = np.linalg.norm(self.accurate_finalstate())
        err = np.sqrt(distance) / magnitude
        abs_err = abs(err)
        return abs_err

    def iteration(self):  # iteration until error is bounded by eps
        circuit1 = self.initialstate()[1]
        if self.error() <= self.eps:
            print("iteration = 1 ", "error = ", self.error())
        else:
            while self.error() >= self.eps:
                circu = circuit1
                self.iterations = self.iterations + 1
                for i in range(self.iterations):
                    gt = self.gate()
                    gt | circu

                a = self.simulation_finalstate()
                b = self.error()

                print(a)
                print("iteration = ", self.iterations, "with error", b)


# example

c = Circuit(4)  # prepare initial state |111>
X | c
i_circuit = c

Hmtn = [[1, 'Z3', 'Z2', 'Z1', 'Z0'],[5, 'Y0']]  # hamiltonian string

Trotter(Hmtn, 1, 0.05, i_circuit=c).Trotter_circuit().draw("command", flatten=True)
print(Trotter(Hmtn, 1, 0.05, i_circuit=c).initialstate()[0])
print(Trotter(Hmtn, 1, 0.05, i_circuit=c).accurate_finalstate())
print(Trotter(Hmtn, 1, 0.05, i_circuit=c).simulation_finalstate())
print(Trotter(Hmtn, 1, 0.05, i_circuit=c).iteration())
