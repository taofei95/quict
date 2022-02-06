from collections import Iterable

from QuICT.core import *
from dag import DAG


class PhasePolynomial:
    """
    Phase polynomial representation of a quantum circuit that only contains
    X, CX and Rz gates.

    DONE circuit -> phase poly
    TODO phase ploy -> circuit
    DONE maybe use DAG?
    """
    def __init__(self, gates):
        """
        Create a phase polynomial from given circuit. The circuit can only contain
        Controlled-X and Rz gates.

        Args:
            gates(Union[CompositeGate, DAG]): Circuit represented by this polynomial
        """

        self.phases = {}
        self.gates = []
        self._build_poly(gates)

    def _build_poly(self, gates):
        monomials = {}
        for gate_ in gates:
            gate_: BasicGate
            for qubit_ in gate_.affectArgs:
                if qubit_ not in monomials:
                    monomials[qubit_] = 1 << (qubit_ + 1)

            if gate_.qasm_name == 'cx':
                monomials[gate_.targ] = monomials[gate_.targ] ^ monomials[gate_.carg]
                self.gates.append(gate_)
            elif gate_.qasm_name == 'x':
                monomials[gate_.targ] = monomials[gate_.targ] ^ 1
                self.gates.append(gate_)
            elif gate_.qasm_name == 'rz':
                sign = -2 * (monomials[gate_.targ] & 1) + 1
                mono = (monomials[gate_.targ] >> 1)
                self.phases[mono] = sign * gate_.parg + (self.phases[mono] if mono in self.phases else 0)
            else:
                raise Exception("PhasePolynomial only accepts cx, x and rz gates.")

    def get_circuit(self):
        """
        Generate a circuit of minimum size that implements the phase polynomial

        Returns:
            CompositeGate: Circuit equivalent to this polynomial
        """
        max_monomial = max(self.phases.keys())
        used_monomials = set()
        circuit_ = []
        for qubit_ in range(int(np.ceil(np.log2(max_monomial)))):
            if (1 << qubit_) in self.phases:
                print(__name__, 'not finished yet')
                break
