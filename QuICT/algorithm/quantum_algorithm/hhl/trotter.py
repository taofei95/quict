from QuICT.core.gate import *


class Trotter(object):
    """ High Order Trotter Decomposition

    References:
        Finding Exponential Product Formulas of Higher Orders: https://arxiv.org/abs/math-ph/0506007v1
    """
    def __init__(self, matrix, order) -> None:
        """
        Args:
            matrix(ndarray): Hermitian matrix(Hamiltonian)
            order(int): the order of trotter decomposition, with error O(t^order)
        """
        self.matrix = matrix
        self.order = order
        assert isinstance(order, int) and order > 0

    def calc_pauli_string(self):
        """ Calculate the coefficients corresponding to all Pauli bases of the matrix

        Return:
            array: index from 0 to len(matrix) ** 2,
                stores the Pauli coefficient of the corresponding index in hexadecimal representation
        """
        m = [[] for _ in range(4)]
        m[0] = np.array([
            [1, 0],
            [0, 1]
        ])
        m[1] = np.array([
            [0, 1],
            [1, 0]
        ])
        m[2] = np.array([
            [0, 0 - 1j],
            [0 + 1j, 0]
        ])
        m[3] = np.array([
            [1, 0],
            [0, -1]
        ])
        theta = []
        matrix = self.matrix
        for idx in range(len(matrix) ** 2):
            s = idx
            p = [1]
            for _ in range(int(np.log2(len(matrix)))):
                p = np.kron(p, m[s % 4])
                s = int(s / 4)
            a = np.trace(np.dot(p, matrix)) / len(matrix)
            theta.append(a)
        return theta

    def trotter_decomposition(self):
        """ decomposite the unitary gate into Clifford-Rz gates

        Return:
            Compositegate
        """
        p = self.p
        gates = CompositeGate()
        control = self.control
        target = self.target
        ancilla = self.ancilla
        for idx in range(len(self.matrix) ** 2):
            st = []
            a = p[idx]
            s = idx
            if abs(a) > 1e-6:
                a /= self.order
                for _ in range(len(target)):
                    st.append(s % 4)
                    s = int(s / 4)
                for c in range(len(target)):
                    if st[c] == 1:
                        H & target[c] | gates
                        CX & [target[c], ancilla] | gates
                    if st[c] == 2:
                        S & target[c] | gates
                        H & target[c] | gates
                        CX & [target[c], ancilla] | gates
                    if st[c] == 3:
                        CX & [target[c], ancilla] | gates
                if control is None:
                    Rz(a) & [ancilla] | gates
                else:
                    for idx in reversed(control):
                        CRz(a) & [idx, ancilla] | gates
                        a *= 2
                for c in reversed(range(len(target))):
                    if st[c] == 1:
                        CX & [target[c], ancilla] | gates
                        H & target[c] | gates
                    if st[c] == 2:
                        CX & [target[c], ancilla] | gates
                        H & target[c] | gates
                        S_dagger & target[c] | gates
                    if st[c] == 3:
                        CX & [target[c], ancilla] | gates
        trotter_gates = CompositeGate()
        for _ in range(self.order):
            gates | trotter_gates
        return trotter_gates

    def excute(self, control, target, ancilla):
        """
        Args:
            control(None/int/list<int>): controlled qubits
            target(int/list<int>): target qubits
            ancilla(None/int): ancilla qubits

        Return:
            np.darray: exactly expm matrix
            CompositeGate: Hamiltonian simulator via trotter decomposition
        """
        self.p = self.calc_pauli_string()
        self.control = control
        self.target = target
        self.ancilla = ancilla
        return self.trotter_decomposition()
