from QuICT.simulation.state_vector import CircuitSimulator
from QuICT.core.gate import *
from QuICT.algorithm.quantum_algorithm.hhl import HHL


class LinearEquation(object):
    def __init__(self, matrix, vector):
        """ linear equation Ax=b

        Args:
            matrix(ndarray): matrix A
            vector(ndarray): vector b

        Raise:
            Exception: The dimensions of A and b are inconsistent
        """
        self.matrix = matrix
        self.vector = vector
        if len(matrix) != len(vector):
            raise Exception("A and b must have same dimension.")

    def solution(self):
        """ solve linear equation by linalg
        """
        return np.linalg.solve(self.matrix, self.vector)

    def hhl(self, t=None, C=None, e=None, measure=None, simulator=None, max_running_time=None):
        """ solve linear equation by HHL algorithm

        Args:
            t: time parameter in C-Unitary Gate
            e(int): number of qubits representing the Phase
            measure(bool): measure ancilla qubit or not
            simulator: CPU or GPU simulator

        Return:
            ndarray: the solution of the linear equation
        """
        if t is None or C is None or e is None:
            evalue, _ = np.linalg.eig(self.matrix)
            if t is None:
                t = 2 * np.pi / max(abs(evalue))
                # t = np.pi * 2 / 3
            if C is None:
                C = 2 * np.arcsin(min(abs(evalue)) * t / (2 * np.pi))
                # C = np.pi/32
            if e is None:
                e = int(np.ceil(np.log2(max(abs(evalue))))) + 1
        if measure is None:
            measure = False
        if simulator is None:
            simulator = CircuitSimulator()
        if max_running_time is None:
            max_running_time = 1

        for _ in range(max_running_time):
            solution = HHL(simulator=simulator).run(
                A=self.matrix,
                b=self.vector,
                t=t,
                C=C,
                p=e,
                measure=measure)
            if solution is not None:
                return solution

    def test(self, measure):
        print('HHL given vector:', self.hhl(measure=measure))
        print('classical state:', self.solution())
