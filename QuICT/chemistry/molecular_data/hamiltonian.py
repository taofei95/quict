#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/9/12 12:09
# @Author  : Xiaoquan Xu
# @File    : moleculardata.py

import itertools
import numpy as np
from numpy import einsum

from QuICT.chemistry.operator.fermion_operator import FermionOperator

def obi_basis_rotation(obi, R):
    """
    Change the basis of a one-body integrals matrix

    Args:
        obi(Ndarray: n*n): one-body integrals of i^ j
        R(Ndarray: n*n): rotation matrix

    Returns:
        Ndarray: n*n one-body integrals of i^ j after rotation
    """
    return einsum("pi,pq,qj->ij", R.conj(), obi, R)

def tbi_basis_rotation(tbi, R):
    """
    Change the basis of a two-body integrals matrix

    Args:
        obi(Ndarray: n*n*n*n): two-body integrals of i^ j^ s t
        R(Ndarray: n*n): rotation matrix

    Returns:
        Ndarray: n*n*n*n two-body integrals of i^ j^ s t after rotation
    """
    return einsum("pi,qj,pquv,us,vt->ijst", R.conj(), R.conj(), tbi, R, R)

class Hamiltonian:
    def __init__(self, const, obi, tbi):
        self.const = const
        self.obi = obi
        self.tbi = tbi

    def get_fermion_operator(self):
        fermion_operator = FermionOperator(0)
        for term in self:
            fermion_operator += FermionOperator(term, self[term])
        return fermion_operator

# The next three functions are still to be completed

    def __iter__(self):

        def sort_key(key):
            """This determines how the keys to n_body_tensors
            should be sorted."""
            # Interpret key as an integer written in binary
            if key == ():
                return 0, 0
            else:
                key_int = int(''.join(map(str, key)))
                return len(key), key_int

        for key in sorted(self.n_body_tensors.keys(), key=sort_key):
            if key == ():
                yield ()
            else:
                n_body_tensor = self.n_body_tensors[key]
                for index in itertools.product(range(self.n_qubits),
                                               repeat=len(key)):
                    if n_body_tensor[index]:
                        yield tuple(zip(index, key))
    
    def __getitem__(self, args):
        """Look up matrix element.

        Args:
            args: Tuples indicating which coefficient to get. For instance,
                `my_tensor[(6, 1), (8, 1), (2, 0)]`
                returns
                `my_tensor.n_body_tensors[1, 1, 0][6, 8, 2]`
        """
        if len(args) == 0:
            return self.n_body_tensors[()]
        else:
            index = tuple([operator[0] for operator in args])
            key = tuple([operator[1] for operator in args])
            return self.n_body_tensors[key][index]

    def __setitem__(self, args, value):
        """Set matrix element.

        Args:
            args: Tuples indicating which coefficient to set.
        """
        if len(args) == 0:
            self.n_body_tensors[()] = value
        else:
            key = tuple([operator[1] for operator in args])
            index = tuple([operator[0] for operator in args])
            self.n_body_tensors[key][index] = value


def generate_hamiltonian(const, obi, tbi, EQ_TOLERANCE=1.0E-12):
    """
        Double the dimension of obi and tbi,
        then generate an interaction operator as Hamiltonian
    """
    n = obi.shape[0]
    n_qubits = 2 * n
    new_obi = np.zeros((n_qubits, n_qubits))
    new_tbi = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits))
    
    for p in range(n):
        for q in range(n):
            new_obi[2*p, 2*q] = obi[p, q]
            new_obi[2*p+1, 2*q+1] = obi[p, q]
    for p in range(n):
        for q in range(n):        
            for r in range(n):
                for s in range(n):
                    # Mixed spin
                    new_tbi[2*p, 2*q+1, 2*r+1, 2*s] = tbi[p, q, r, s] / 2.
                    new_tbi[2*p+1, 2*q, 2*r, 2*s+1] = tbi[p, q, r, s] / 2.

                    # Same spin
                    new_tbi[2*p, 2*q, 2*r, 2*s] = tbi[p, q, r, s] / 2.
                    new_tbi[2*p+1, 2*q+1, 2*r+1, 2*s+1] = tbi[p, q, r, s] / 2.

    # Truncate.
    new_obi[np.absolute(new_obi) < EQ_TOLERANCE] = 0.
    new_tbi[np.absolute(new_tbi) < EQ_TOLERANCE] = 0.

    return Hamiltonian(const, new_obi, new_tbi)
