from QuICT.simulation.state_vector import StateVectorSimulator
from QuICT.core.gate import *
from QuICT.algorithm.quantum_algorithm.hhl import HHL


class LinearEquation(object):
    def __init__(self, matrix, vector):
        """ linear equation Ax=b,
            where A is the given matrix and b is the given vector

        Args:
            matrix(ndarray/matrix): matrix A
            vector(array): vector b

        Raise:
            Exception: The dimensions of A and b are inconsistent
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

    def hhl(self, t=None, e=None, method=None, measure=None, order=None, simulator=None):
        """ solving linear equation by HHL algorithm

        Args:
            t(float): the coefficient makes matrix (t*A/2pi)'s eigenvalues are in (1/e, 1)
            e(int): number of qubits representing the Phase
            method: method is Hamiltonian simulation, default "trotter"
            measure(bool): measure the ancilla qubit or not
            order(int): trotter number if method is "trotter"
            simulator: CPU or GPU simulator

        Return:
            array: the solution of the linear equation
            "Failedd.": the solution is none
        """
        if t is None or e is None:
            evalue = np.linalg.eigvals(self.matrix)
            if t is None:
                t = np.pi / max(abs(evalue))
            if e is None:
                e = int(np.ceil(np.log2(max(abs(evalue)) / min(abs(evalue))))) + 2
        if method is None:
            method = 'trotter'
            if order is None:
                order = 1
        if measure is None:
            measure = False
        if simulator is None:
            simulator = StateVectorSimulator()

        solution = HHL(simulator=simulator).run(
            matrix=self.matrix,
            vector=self.vector,
            t=t,
            e=e,
            method=method,
            measure=measure,
            order=order)
        if solution is not None:
            return solution
        else:
            return "Failed."
