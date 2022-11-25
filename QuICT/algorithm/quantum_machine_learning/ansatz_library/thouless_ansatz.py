#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2022/4/26 8:33
# @Author  : Ou Shigang
# @File    : ansatz.py

from QuICT.core.gate import CompositeGate
from QuICT.core import Circuit
from QuICT.core.gate import Rz, sqiSwap, X

import numpy as np


class Thouless():
    """ Thouless ansatz implemented from 

    Hartree-Fock on a superconducting qubit quantum computer[Google Inc.]
    
    arXiv: 2004.04174v4 [quant-ph] 18 Sep 2020
    
    """
    @staticmethod
    def modified_Givens_rotation(angle):
        cgate = CompositeGate()

        with cgate:
            sqiSwap & [0, 1]
            Rz(angle) & 0
            Rz(np.pi - angle) & 1
            sqiSwap & [0, 1]
            Rz(np.pi) & 0
        
        return cgate

    def build_circuit(n, angles, num_electron_pairs):
        '''Quantum Circuits with n qubits and C(n 2)  R(theta)[p,q] gates
        
        Args:
            n(int): number of orbits (e.g. quantum qubits)
            angles(List[float]): a list with C(N 2) parameter
            num_electron_pairs(int): number of electron pairs

        Returns:
            Circuit: Quantum Circuits with N qubits and C(N 2) R gates
        '''
        # circuits with n qubits
        circuit = Circuit(n)

        gate = Thouless.build_gate(n, angles, num_electron_pairs)

        gate | circuit
        
        return circuit

    def build_gate(n, angles, num_electron_pairs):
        """ 
        build a compositegate from a list of parameters

        Args:
            args(tuple): parameters of the ansatz builder
        
        Returns:
            (CompositeGate): a composite gate built from the given arguments

        """
        # circuits with n qubits
        ansatz = CompositeGate()

        R = Thouless.modified_Givens_rotation

        modified = {}

        with ansatz:
            temp = num_electron_pairs
            while temp > 0:
                X & [temp - 1]
                modified[temp - 1] = 1
                temp -= 1
            # add gates to the circuits in parallelization
            if(n == 1):
                return ansatz
            elif(n == 2):
                R(angles.pop()) & [0, 1] # assign the parameters to each circuit
            elif(n > 2):
                i = 0

                while i < n - 1:
                    layer = i + 1 # number of layer
                    gate = 0 # number of gates
                    index = i
                    while index >= 0:
                        k = n - index - 2
                        if  (k in modified or k + 1 in modified) and gate < layer and gate < n - min(n, num_electron_pairs): # if any qubit is modified
                            modified[k] = 1
                            modified[k+1] = 1
                            R(angles.pop()) & [k, k + 1]
                            gate += 1
                        index -= 2
                    i += 1

                while i > 2:
                    gate = 0
                    layer = i - min(n, num_electron_pairs)
                    index = i-1
                    while index >= 2 and gate < min(n, num_electron_pairs) and gate < layer:
                        k = n - index - 2
                        R(angles.pop()) & [k, k + 1]
                        index -= 2
                        gate += 1
                    i -= 1
            
        return ansatz