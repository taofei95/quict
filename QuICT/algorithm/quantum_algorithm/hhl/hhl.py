import logging

from QuICT.qcda.synthesis.unitary_decomposition import *
from QuICT.simulation.state_vector import CircuitSimulator
from QuICT.qcda.synthesis.quantum_state_preparation import *
from QuICT.algorithm import Algorithm
from QuICT.core import *
from QuICT.core.gate import *
from scipy.linalg import expm


class HHL(Algorithm):
    def __init__(self, simulator=CircuitSimulator()) -> None:
        self.simulator = simulator

    @staticmethod
    def resize(matrix, vector):
        row, column = matrix.shape
        if row != column:
            return Exception("A must be a square matrix")
        if row != len(vector):
            raise Exception("A and b must have same number of rows!")
        n = int(2 ** np.ceil(np.log2(row)))

        H = np.identity(n, dtype=np.complex128)
        H[:row, :column] = matrix.copy()
        if (H == H.T.conj()).all():
            pass
        else:
            H2 = np.zeros((2 * n, 2 * n), dtype=np.complex128)
            H2[:n, n:] = H.copy()
            H2[n:, :n] = H.T.conj().copy()
            H = H2.copy()

        rho = np.zeros(len(H))
        rho[len(H) - len(vector):] = vector.copy()

        return H, rho

    @staticmethod
    def construct_QPE(control, target, A, t):
        targ = target[0]
        n, _ = A.shape
        # t /= 2 ** len(target)
        gates = CompositeGate()
        with gates:
            for idx in control:
                H & idx
            for idx in control:
                U = expm(A * t * 1j)
                t /= 2.0
                ugate, _ = UnitaryDecomposition().execute(U)
                for i in range(ugate.size()):
                    gate = ugate[i]
                    if gate.type == GateType.unitary:
                        u = gate.matrix
                        u = u.reshape(2, 2)
                        eps = 1e-6
                        a = np.angle(u[0, 0])
                        alpha = np.exp(1j * a)
                        u = u / alpha
                        theta = np.arccos(u[0, 0])
                        sint = np.sin(theta)
                        if abs(sint) >= eps:
                            lamda = np.angle(u[0, 1] / -sint)
                            phi = np.angle(u[1, 0] / sint)
                        else:
                            lamda = 0
                            phi = np.angle(u[1, 1] / np.cos(theta))

                        CU3(theta * 2, phi, lamda) & [idx, targ + gate.targ]
                        if abs(a) >= eps:
                            U1(a) & idx
                    elif gate.type == GateType.cx:
                        CCX & [idx, targ + gate.carg, targ + gate.targ]
                    elif gate.type == GateType.rz:
                        CRz(gate.parg) & [idx, targ + gate.targ]
                    elif gate.type == GateType.ry:
                        CU3(gate.parg, 0, 0) & [idx, targ + gate.targ]
                    elif gate.type == GateType.gphase:
                        if abs(gate.parg) >= eps:
                            U1(gate.parg) & idx
                    else:
                        print(gate.type)
                        raise Exception('Unknown gate')
        return gates

    @staticmethod
    def rotation(circuit, control, target, theta):
        # not optimal
        for idx in target:
            CU3(theta, 0, 0) | circuit([idx, control])
            theta /= 2

    @staticmethod
    def solve(state_vector, matrix, vector):
        eps = 1e-8
        if state_vector is not None:
            for idx, x in enumerate(vector):
                if abs(x) > eps:
                    k = np.vdot(matrix[idx], state_vector)
                    if abs(k) > eps:
                        return state_vector * x / k

    def circuit(self, A, b, t, C, p, measure=True):
        """ hhl algorithm to solve linear equation such as Ax=b

        Args:
            A(array/matrix): the matrix A above
                [A must be reversible]
            b(ndarray): the vector b above, need to be prepared previously
                A and b MUST have same number of ROWS!
            t(float): parameter to simulate Hamiltonian e^(iAt)
            C(float): parameter in Controlled-Rotate gates
            p(int): use p qubits to express estimated phase
            measure(bool): Measure ancilla qubit or not

        Returns:
            ndarray: the goal state vector if algorithm success
            None: algorithm failed
        """
        n = int(np.log2(len(A)))
        circuit = Circuit(n + p + 1)
        ancilla = 0
        phase = list(range(1, p + 1))
        x = list(range(p + 1, p + n + 1))

        # Prepare
        if len(x) > 1:
            QuantumStatePreparation().execute(b) | circuit(x)
        else:
            U3(2 * np.arcsin(b[1]), 0, 0) | circuit(x)

        # QPE
        QPE = self.construct_QPE(phase, x, A, t)
        QPE | circuit
        IQFT.build_gate(len(phase)) | circuit(list(reversed(phase)))

        # Rotation
        self.rotation(circuit, ancilla, phase, C)

        # inverse QPE
        QFT.build_gate(len(phase)) | circuit(list(reversed(phase)))
        QPE.inverse() | circuit

        # Simulation

        if measure:
            Measure | circuit(ancilla)

        logging.info(
            f"circuit width    = {circuit.width():4}" +
            f"circuit size     = {circuit.size():4}"
        )

        return circuit

    def run(self, A, b, t, C, p, measure=True):
        simulator = self.simulator
        size = len(b)
        A, b = self.resize(A, b)
        n = int(np.log2(len(A)))
        circuit = self.circuit(A, b / np.sqrt(np.vdot(b, b)), t, C, p, measure)
        state_vector = simulator.run(circuit)
        x = state_vector[2 ** (n + p): 2 ** (n + p) + size]
        if not measure or int(circuit[0]) == 1:
            if x is not None:
                return self.solve(x, A, b)
