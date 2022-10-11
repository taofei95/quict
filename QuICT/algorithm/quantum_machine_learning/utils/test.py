import numpy as np
import torch
import copy

def pauli_words_matrix(pauli_words_r) -> list:
        def to_matrix(string):
            matrix = torch.tensor([1 + 0j], dtype=torch.complex128)
            for i in range(len(string)):
                if string[i] == 'X':
                    matrix = torch.kron(
                        matrix,
                        torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
                    )
                elif string[i] == 'Y':
                    matrix = torch.kron(
                        matrix,
                        torch.tensor([[0j, -1j], [1j, 0j]], dtype=torch.complex128)
                    )
                elif string[i] == 'Z':
                    matrix = torch.kron(
                        matrix,
                        torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
                    )
                elif string[i] == 'I':
                    matrix = torch.kron(
                        matrix,
                        torch.tensor([[1, 0], [0, 1]], dtype=torch.complex128)
                    )
                else:
                    raise ValueError('wrong format of string ' + string)
            return matrix

        return list(map(to_matrix, copy.deepcopy(pauli_words_r)))
    
pauli_words_r = ['ZZ', 'ZZ']
a = pauli_words_matrix(pauli_words_r)
print(a)
