from QuICT.tools import Logger
from QuICT.qcda.synthesis.quantum_state_preparation import QuantumStatePreparation
from QuICT.algorithm.quantum_algorithm.hhl.trotter import Trotter
from QuICT.qcda.synthesis.unitary_decomposition.controlled_unitary import ControlledUnitaryDecomposition
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
            raise Exception("A and b must have same length!")
        n = int(2 ** np.ceil(np.log2(row)))

        m = np.identity(n, dtype=np.complex128)
        m[n - row:, n - row:] = matrix
        v = np.zeros(n, dtype=np.complex128)
        v[n- len(vector):] = vector
        if not (m == m.T.conj()).all():
            m0 = np.zeros((2 * n, 2 * n), dtype=np.complex128)
            m0[:n, n:] = m
            m0[n:, :n] = m.T.conj()
            m = m0
            v0 = np.zeros(2 * n, dtype=np.complex128)
            v0[:n] = v
            v = v0
        return m, v

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
        c = 1
        for l in range(1, 2 ** len(control)):
            for idx in range(len(control)):
                if ((l >> idx) & 1) == 0:
                    X & [control[idx]] | gates
            for idx in range(len(ancilla)):
                if idx == 0:
                    CCX & [control[0], control[1], ancilla[0]] | gates
                else:
                    CCX & [control[idx + 1], ancilla[idx - 1], ancilla[idx]] | gates
            if l < 2 ** (len(control) - 1):
                CU3(2 * np.arcsin(c / l), 0, 0) & [ancilla[-1], target] | gates
            else:
                CU3(2 * np.arcsin(c / (l - 2 ** len(control))), 0, 0) & [ancilla[-1], target] | gates
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
                    k = np.vdot(state_vector, matrix[idx])
                    if abs(k) > eps:
                        return state_vector * x / k

    def circuit(self, matrix, vector, t, e, measure=True, method='unitary'):
        """
        Args:
            matrix(ndarray/circuit): the matrix A above
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
        if isinstance(method, tuple):
            if method[0] == 'trotter':
                circuit = Circuit(n + 2 * e + 1)
                ancilla = 0
                phase = list(range(1, e + 1))
                toffoli = list(range(e + 1, 2 * e))
                trotter = 2 * e
                x = list(range(2 * e + 1, 2 * e + n + 1))
                order = method[1]
                method = method[0]
        elif method == 'unitary':
            circuit = Circuit(2 * e + n)
            ancilla = 0
            phase = list(range(1, e + 1))
            toffoli = list(range(e + 1, 2 * e))
            x = list(range(2 * e, 2 * e + n))

        if len(x) > 1:
            QuantumStatePreparation().execute(vector) | circuit(x)
        else:
            Ry(2 * np.arcsin(vector[1])) | circuit(x)

        if method == 'trotter':
            CU = Trotter(matrix * t, order).excute(phase, x, trotter)
        elif method == 'unitary':
            from scipy.linalg import expm
            CU = CompositeGate()
            for idx in reversed(phase):
                c = [idx]
                c.extend(x)
                U, _ = ControlledUnitaryDecomposition().execute(np.identity(2 ** n, dtype=np.complex128), expm(matrix * t * 0.5j))
                U | CU(c)
                t *= 2.0
        else:
            raise Exception(f"unknown method: {method}")

        for idx in phase:
            H | circuit(idx)
        CU | circuit
        IQFT.build_gate(len(phase)) | circuit(list(reversed(phase)))

        CRY = self.c_rotation(phase, ancilla, toffoli)
        Ry(-0.9 * np.pi) | circuit(ancilla)
        CRY | circuit

        QFT.build_gate(len(phase)) | circuit(list(reversed(phase)))
        CU.inverse() | circuit

        for idx in phase:
            H | circuit(idx)
        
        if measure:
            Measure | circuit(ancilla)

        self.hs = CU.size()
        self.hd = CU.depth()
        logger.info(
            f"circuit width    = {circuit.width():4}\n" +
            f"circuit size     = {circuit.size():4}\n" +
            f"hamiltonian size = {CU.size():4}\n" +
            f"CRy size         = {CRY.size():4}"
        )

        return circuit

    def run(self, matrix, vector, t, e, measure=True, method='unitary'):
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
        m, v = self.reconstruct(matrix, vector)
        nv = np.linalg.norm(v)
        circuit = self.circuit(m, v / nv, t, e, measure, method)

        import time
        rt = time.time()
        state_vector = simulator.run(circuit)
        rt = round(time.time() - rt, 6)
        x = np.array(state_vector[len(v) - size: len(v)].get(), dtype=np.complex128)
        a0 = np.array(state_vector[:len(state_vector) / 2].get())
        a0 = np.vdot(a0, a0)
        a1 = np.array(state_vector[len(state_vector) / 2:].get())
        a1 = np.vdot(a1, a1)
        print(f"p(0) = {a0}; p(1) = {a1}")
        msg = [circuit.width(), circuit.size(),self.hs, rt]

        if not measure or int(circuit[0]) == 0:
            return x, msg
        else:
            return None, msg