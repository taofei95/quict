from QuICT.simulation.state_vector import StateVectorSimulator
from QuICT.core.gate import *
from QuICT.algorithm.quantum_algorithm.hhl import HHL


class LinearSolver(object):
    def __init__(self, matrix, vector):
        """ linear equation Ax=b,
            where A is the given matrix and b is the given vector

        Args:
            matrix(ndarray/matrix): matrix A
            vector(array): vector b
        """
        self.matrix = matrix
        self.vector = vector
        assert len(matrix) == len(vector)

    def solution(self):
        """
        Return:
            array: solution from classical linalg
        """
        return np.linalg.solve(self.matrix, self.vector)

    def hhl(self, t=None, e=None, method=None, simulator=None):
        """ solving linear equation by HHL algorithm

        Args:
            t(float): the coefficient makes matrix (t*A/2pi)'s eigenvalues are in (1/2^e, 1)
            e(int): number of qubits representing the Phase
            method: Hamiltonian simulation method, default "trotter"
            simulator: CPU or GPU simulator

        Return:
            Tuple[array, 'Failed']
                array: the solution of the linear equation
                "Failed.": the solution is none
        """
        if t is None or e is None:
            evalue = np.linalg.eigvals(self.matrix)
            if t is None:
                t = 0.5 * np.pi / max(abs(evalue))
            if e is None:
                e = 9
        if method is None:
            method = 'unitary'
        if simulator is None:
            simulator = StateVectorSimulator(device="GPU")

        solution = HHL(simulator=simulator).run(
            matrix=self.matrix,
            vector=self.vector,
            t=t,
            e=e,
            method=method)
        return solution
