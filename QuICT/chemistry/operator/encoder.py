#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/8/17 17:30
# @Author  : Xiaoquan Xu
# @File    : encoder.py

"""
Encoders transform ladder operators to CompositeGate on zero state of qubits.
"""

from QuICT.chemistry.operator.qubit_operator import QubitOperator

class Encoder:
    """
    Superclass of encoding methods.

    Attributes:
        n_orbitals_default(integer): the default number of orbitals
    """
    n_orbitals_default = 10

    @classmethod
    def encoder(cls, fermion_operator, n_orbitals = -1):
        """
        Encoders transform ladder operators to Qubit Operators.

        Args:
            fermion_operator(FermionOperator): FermionOperator to be transformed
            n_orbitals(integer): the number of all the orbitals

        Returns:
            QubitOperator: The corresponding operators on qubits
        """
        if n_orbitals == -1:
            n_orbitals = Encoder.n_orbitals_default
        ans = QubitOperator(0)
        for mono_f in fermion_operator.operators:
            mono_q = QubitOperator([], mono_f[1])
            for operator in mono_f[0]:
                mono_q *= cls.encoder_single(operator[0], operator[1], n_orbitals)
            ans += mono_q
        return ans
    
    @classmethod
    def encoder_single(cls, target, kind, n_orbitals):
        """
        Encode a single ladder operator (To be overrided).

        Args:
            target(integer): the target fermion of the single ladder operator (>=0)
            kind(integer): whether the operator is annihilation(1) or creation(0)
            n_orbitals(integer): the number of all the orbitals

        Returns:
            QubitOperator: The corresponding operators on qubits
        """
        raise Exception("The encoder is not realized.")
    
def trans_01(target):
    """
    Construct the operator |0><1| on the target qubit.
    The next three functions are similar.

    Args:
        target(integer): index of the target qubit (>=0)
    
    Returns:
        QubitOperator: '0.5X+0.5iY' on the target qubit
    """
    return QubitOperator([(target, 1)], 0.5) + QubitOperator([(target, 2)], 0.5j)

def trans_10(target):
    return QubitOperator([(target, 1)], 0.5) + QubitOperator([(target, 2)], -0.5j)

def trans_00(target):
    return QubitOperator([], 0.5) + QubitOperator([(target, 3)], 0.5)

def trans_11(target):
    return QubitOperator([], 0.5) + QubitOperator([(target, 3)], -0.5)

class JordanWigner(Encoder):
    """
    Implement the Jordan-Wigner encoding method.
    """         
    @classmethod
    def encoder_single(cls, target, kind, n_orbitals):
        Zlist = [(i,3) for i in range(target)]
        ans = QubitOperator(Zlist)
        if kind == 0:               #annihilation
            ans *= trans_01(target)
        else:                       #creation
            ans *= trans_10(target)
        return ans

class Parity(Encoder):
    """
    Implement the parity encoding method.
    """
    @classmethod
    def encoder_single(cls, target, kind, n_orbitals):
        Xlist = [(i,1) for i in range(target + 1, n_orbitals)]
        ans = QubitOperator(Xlist)
        if kind == 0:               #annihilation
            if target == 0:
                ans *= trans_01(target)
            else:
                ans *= trans_00(target - 1) * trans_01(target) - trans_11(target - 1) * trans_10(target)
        else:                       #creation
            if target == 0:
                ans *= trans_10(target)
            else:
                ans *= trans_00(target - 1) * trans_10(target) - trans_11(target - 1) * trans_01(target)
        return ans

def flip(x, n_orbitals):
    """
    Construct a list of indexes involved in flipping n_x.

    Args:
        x(integer): index of the target qubit (x>=0)
        n_orbitals(integer): the number of all the orbitals
    
    Returns:
        list: the indexes involved in flipping n_x
    """
    index = []
    x += 1
    while (x <= n_orbitals):
        index.append(x - 1)
        x += x & (-x)
    return index

def sumup(x):
    """
    Construct a list of indexes involved in summation n_0, ..., n_x

    Args:
        x(integer): index of the target qubit. (x>=0)
            Specially, if x is -1, return []
    
    Returns:
        list: the indexes involved in summation n_0, ..., n_x
    """
    index = []
    x += 1
    while (x > 0):
        index.append(x - 1)
        x -= x & (-x)
    return index[::-1]

class BravyiKitaev(Encoder):
    """
    Implement the Bravyi-Kitaev encoding method
    """
    @classmethod
    def encoder_single(cls, target, kind, n_orbitals):
        Xlist = [(i,1) for i in flip(target, n_orbitals)]
        ans = QubitOperator(Xlist, 0.5)
        Zlist1 = [(i,3) for i in sumup(target - 1)]
        Zlist2 = [(i,3) for i in sumup(target)]
        ans *= QubitOperator(Zlist1) - QubitOperator(Zlist2, (-1)**kind)
        return ans
