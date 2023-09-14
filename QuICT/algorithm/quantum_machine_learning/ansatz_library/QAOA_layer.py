from typing import Union

import numpy as np

from QuICT.algorithm.quantum_machine_learning.utils import Hamiltonian
from QuICT.core import Circuit
from QuICT.core.gate import CX, GPhase, H, Hy, Rx, Ry, Rz, Variable
from QuICT.tools.exception.algorithm import *

from .ansatz import Ansatz


class QAOALayer(Ansatz):
    """The quantum approximate optimization algorithm (QAOA) ansatz."""

    def __init__(
        self, n_qubits: int, p: int, hamiltonian: Hamiltonian,
    ):
        """Initialize a QAOA ansatz instance.

        Args:
            n_qubits (int): The number of qubits.
            p (int): The number of layers of the QAOA ansatz.
            hamiltonian (Hamiltonian): The hamiltonian for a specific combinatorial-optimization problem.
        """
        super(QAOALayer, self).__init__(n_qubits)
        self._p = p
        self._hamiltonian = hamiltonian

    def init_circuit(self, params: Union[Variable, np.ndarray] = None):
        """Initialize a QAOA ansatz with trainable parameters.

        Args:
            params (Union[Variable, np.ndarray], optional): Initialization parameters. Defaults to None.

        Returns:
            Circuit: The QAOA ansatz.
        """
        params = np.random.randn(2, self._p) if params is None else params
        params = Variable(pargs=params) if isinstance(params, np.ndarray) else params
        if params.shape == (2, self._p):
            self._params = params
        else:
            raise AnsatzShapeError(str((2, self._p)), str(params.shape))

        circuit = Circuit(self._n_qubits)
        # initialize state vector
        H | circuit

        for p in range(self._p):
            # construct U_gamma
            U_gamma = self._construct_U_gamma_circuit(self._params[0][p])
            U_gamma | circuit(list(range(self._n_qubits)))

            # construct U_beta
            U_beta = Rx(2 * self._params[1][p])
            U_beta | circuit

        circuit.gate_decomposition(decomposition=False)
        return circuit

    def _construct_U_gamma_circuit(self, gamma):
        circuit = Circuit(self._n_qubits)
        for coeff, qids, gates in zip(
            self._hamiltonian._coefficients,
            self._hamiltonian._qubit_indexes,
            self._hamiltonian._pauli_gates,
        ):
            gate_dict = {
                "X": {"mqids": H, "qid": Rx(2 * coeff * gamma)},
                "Y": {"mqids": Hy, "qid": Ry(2 * coeff * gamma)},
                "Z": {"qid": Rz(2 * coeff * gamma)},
            }

            # Mapping e.g. Rxyz
            if len(qids) > 1:
                for i in range(len(qids)):
                    if gates[i] != "Z":
                        gate_dict[gates[i]]["mqids"] | circuit(qids[i])
                Rnz_circuit = self._Rnz_circuit(2 * coeff * gamma, qids)
                Rnz_circuit | circuit(list(range(self._n_qubits)))
                for i in range(len(qids)):
                    if gates[i] != "Z":
                        gate_dict[gates[i]]["mqids"] | circuit(qids[i])

            # Only Rx, Ry, Rz
            elif len(qids) == 1:
                gate_dict[gates[0]]["qid"] | circuit(qids[0])

            # Only coeff
            else:
                GPhase | circuit

        return circuit

    def _Rnz_circuit(self, gamma, tar_idx: Union[int, list]):
        circuit = Circuit(self._n_qubits)
        if isinstance(tar_idx, int):
            Rz(gamma) | circuit(tar_idx)
        else:
            # Add CNOT gates
            for i in range(len(tar_idx) - 1):
                CX | circuit(tar_idx[i: i + 2])
            # Add RZ gate
            Rz(gamma) | circuit(tar_idx[-1])
            # Add CNOT gates
            for i in range(len(tar_idx) - 2, -1, -1):
                CX | circuit(tar_idx[i: i + 2])
        return circuit
