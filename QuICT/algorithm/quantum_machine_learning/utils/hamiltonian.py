from typing import Dict, List, Union
import numpy as np


class Hamiltonian:
    def __init__(self, hamiltonian: Union[np.ndarray, list]):
        self._hamiltonian = hamiltonian
        self._matrix_h = None

        if isinstance(self._hamiltonian, np.ndarray):
            self._matrix_h = hamiltonian
            self._hamiltonian = self.matrix2paulistr()

    def matrix2paulistr(self):
        return self._jordan_wigner(self._hamiltonian)

    def paulistr2matrix(self):
        return

