from typing import Dict, List, Union
import numpy as np
import copy


class Hamiltonian:
    @property
    def coefficients(self):
        return self._coefficients

    @property
    def pauli_gates(self):
        return self._pauli_gates

    @property
    def qubit_indexes(self):
        return self._qubit_indexes

    def __init__(self, pauli_str: list):
        self._pauli_str = pauli_str
        self._coefficients = []
        self._pauli_gates = []
        self._qubit_indexes = []

        self._pauli_str_validation()

    def __getitem__(self, indexes):
        pauli_str = []
        if isinstance(indexes, int):
            indexes = [indexes]
        for idx in indexes:
            pauli_str.append(self._pauli_str[idx])
        return Hamiltonian(pauli_str)

    def __add__(self, other):
        return Hamiltonian(self._pauli_str + other._pauli_str)

    def __sub__(self, other):
        return self.__add__(other.__mul__(-1))

    def __mul__(self, other: float):
        new_pauli_str = copy.deepcopy(self.pauli_str)
        for pauli_operator in new_pauli_str:
            pauli_operator[0] *= other
        return Hamiltonian(new_pauli_str)

    def _pauli_str_validation(self):
        for pauli_operator in self._pauli_str:
            self._pauli_operator_validation(pauli_operator)

    def _pauli_operator_validation(self, pauli_operator):
        assert isinstance(pauli_operator[0], int) or isinstance(
            pauli_operator[0], float
        ), "The coefficient of a Pauli operator must be integer or float."
        self._coefficients.append(float(pauli_operator[0]))

        assert len(pauli_operator) == len(
            set(pauli_operator)
        ), "The qubit indice of the same Pauli Gate cannot be the same."

        indexes = []
        pauli_gates = ""
        for pauli_gate in pauli_operator[1:]:
            assert (
                pauli_gate[0] in ["X", "Y", "Z", "I"] and pauli_gate[1:].isdigit()
            ), "The Pauli gate should be in the format of gate + index。 e.g. Z0, I5, Y3."
            if pauli_gate[0] == "I":
                continue
            pauli_gates += pauli_gate[0]
            indexes.append(int(pauli_gate[1:]))
        self._qubit_indexes.append(indexes)
        self._pauli_gates.append(pauli_gates)


def main():
    pauli_str = [[1, "Z0", "I2"], [0.05, "Z1", "Y2", "X4"]]
    h = Hamiltonian(pauli_str)
    print(h._pauli_gates)
    print(h._qubit_indexes)


if __name__ == "__main__":
    main()

