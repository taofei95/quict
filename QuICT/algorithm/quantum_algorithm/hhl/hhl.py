from QuICT.tools import Logger
from QuICT.qcda.synthesis.unitary_decomposition import UnitaryDecomposition
from QuICT.qcda.synthesis.quantum_state_preparation import QuantumStatePreparation
from QuICT.simulation.state_vector import StateVectorSimulator
from QuICT.algorithm.quantum_algorithm.hhl.trotter import Trotter
from QuICT.core import Circuit
from QuICT.core.gate import *


logger = Logger('hhl')


class HHL:
    """ original HHL algorithm

    References:
        [1] Quantum Algorithm for Linear Systems of Equations: https://doi.org/10.1103/PhysRevLett.103.150502
        [2] Quantum circuit design for solving linear systems of equations: https://doi.org/10.1080/00268976.2012.668289
    """
    def __init__(self, simulator) -> None:
        self.simulator = simulator

    def reconstruct(self, matrix, vector):
        """ When matrix A is not Hermitian, construct a hermite matrix H, and A is in the upper right corner.
            Simultaneously, vector b add 0 in the tail.
        """
        row, column = matrix.shape
        if row != column:
            return Exception("A must be a square matrix")
        if row != len(vector):
            raise Exception("A and b must have same number of rows!")
        n = int(2 ** np.ceil(np.log2(row)))

        H = np.identity(n, dtype=np.complex128)
        H[:row, :column] = matrix
        if not (H == H.T.conj()).all():
            H2 = np.zeros((2 * n, 2 * n), dtype=np.complex128)
            H2[:n, n:] = H
            H2[n:, :n] = H.T.conj()
            H = H2
        rho = np.zeros(len(H), dtype=np.complex128)
        rho[len(H) - len(vector):] = vector
        return H, rho

    def c_rotation(self, control, target, ancilla):
        """Controlled-Rotation part in HHL algorithm

        Args:
            control(int/list[int]): control qubits in multicontrol toffoli gates, in HHL it is phase qubits part
            target(int): target qubit operate CRy gate, in HHL it is ancilla qubit part
            ancilla(int/list[int]): ancilla qubits in multicontrol toffoli gates,
                                    in HHL it needs additional auxiliary qubits

        Return:
            CompositeGate
        """
        gates = CompositeGate()
        c = 0.5
        for l in range(1, 2 ** len(control)):
            for idx in range(len(control)):
                if ((l >> idx) & 1) == 0:
                    X & [control[idx]] | gates
            for idx in range(len(ancilla)):
                if idx == 0:
                    CCX & [control[0], control[1], ancilla[0]] | gates
                else:
                    CCX & [control[idx + 1], ancilla[idx - 1], ancilla[idx]] | gates
            CU3(2 * np.arcsin(c / l), 0, 0) & [ancilla[-1], target] | gates
            for idx in reversed(range(len(ancilla))):
                if idx == 0:
                    CCX & [control[0], control[1], ancilla[0]] | gates
                else:
                    CCX & [control[idx + 1], ancilla[idx - 1], ancilla[idx]] | gates
            for idx in range(len(control)):
                if ((l >> idx) & 1) == 0:
                    X & control[idx] | gates
        X & target | gates
        return gates

    def solve(self, state_vector, matrix, vector):
        eps = 1e-8
        if state_vector is not None:
            for idx, x in enumerate(vector):
                if abs(x) > eps:
                    k = np.vdot(matrix[idx], state_vector)
                    if abs(k) > eps:
                        return state_vector * x / k

    def circuit(self, matrix, vector, t, e, measure=True, method='trotter', order=1):
        """
        Args:
            matrix(ndarray/matrix): the matrix A above
                [A must be invertible]
            vector(array): the vector b above, need to be prepared previously
                matrix and vector MUST have the same number of ROWS!
            t(float): the coefficient makes matrix (t*A/2pi)'s eigenvalues are in (1/e, 1)
            e(int): number of qubits representing the Phase
            method: method is Hamiltonian simulation, default "trotter"
            measure(bool): measure the ancilla qubit or not
            order(int): trotter number if method is "trotter"

        Returns:
            Circuit: HHL circuit
        """
        n = int(np.log2(len(matrix)))
        if method == 'trotter':
            circuit = Circuit(n + 2 * e + 1)
            ancilla = 0
            phase = list(range(1, e + 1))
            toffoli = list(range(e + 1, 2 * e))
            trotter = 2 * e
            x = list(range(2 * e + 1, 2 * e + n + 1))
        else:
            circuit = Circuit(n + e + 1)
            ancilla = 0
            phase = list(range(1, e + 1))
            x = list(range(e + 1, e + n + 1))

        if len(x) > 1:
            QuantumStatePreparation().execute(vector) | circuit(x)
        else:
            Ry(2 * np.arcsin(vector[1])) | circuit(x)

        if method == 'trotter':
            CU = Trotter(matrix * t, order).excute(phase, x, trotter)
        elif method == 'unitary':
            CU = self.unitary_decomposition(phase, x, matrix, t)
        else:
            raise Exception(f"unknown method: {method}")
        for idx in phase:
            H | circuit(idx)
        CU | circuit
        IQFT.build_gate(len(phase)) | circuit(list(reversed(phase)))

        CRY = self.c_rotation(phase, ancilla, toffoli)
        CRY | circuit

        QFT.build_gate(len(phase)) | circuit(list(reversed(phase)))
        CU.inverse() | circuit

        for idx in phase:
            H | circuit(idx)
        if measure:
            Measure | circuit(ancilla)

        logger.info(
            f"circuit width    = {circuit.width():4}\n" +
            f"circuit size     = {circuit.size():4}\n" +
            f"hamiltonian size = {CU.size():4}\n" +
            f"CRy size         = {CRY.size():4}"
        )

        return circuit

    def run(self, matrix, vector, t, e, measure=True, method='trotter', order=1):
        """ hhl algorithm to solve linear equation such as Ax=b,
            where A is the given matrix and b is the given vector

        Args:
            matrix(ndarray/matrix): the matrix A above
                [A must be reversible]
            vector(array): the vector b above, need to be prepared previously
                matrix and vector MUST have the same number of ROWS!
            t(float): the coefficient makes matrix (t*A/2pi)'s eigenvalues are in (1/e, 1)
            e(int): number of qubits representing the Phase
            method: method is Hamiltonian simulation, default "trotter"
            measure(bool): measure the ancilla qubit or not
            order(int): trotter number if method is "trotter"

        Returns:
            array: the goal state vector if the algorithm success
            None: algorithm failed
        """
        simulator = self.simulator
        size = len(vector)
        m = matrix.copy()
        v = vector.copy()
        matrix, vector = self.reconstruct(matrix, vector)
        nv = np.linalg.norm(vector)
        circuit = self.circuit(matrix, vector / nv, t, e, measure, method, order)

        state_vector = simulator.run(circuit)

        x = state_vector[: size]

        if not measure or int(circuit[0]) == 0:
            return self.solve(x, m, v)
