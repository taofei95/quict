#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2022/4/26 8:33
# @Author  : Ou Shigang
# @File    : ansatz.py

from QuICT.core.gate import CompositeGate
from QuICT.core import Circuit
from QuICT.core.gate import Rz, sqiSwap

import numpy as np

class Ansatz(object):
    """ ansatz used to generate composite gates"""

    def build_gate(self, *args):
        """ 
        build a compositegate from a list of parameters

        Args:
            args(tuple): parameters of the ansatz builder
        
        Returns:
            (CompositeGate): a composite gate built from the given arguments

        """
        ...
    
    def build_circuit(self, *args):
        """ 
        build a quantum circuit from a list of parameters

        Args:
            args(tuple): parameters of the ansatz builder
        
        Returns:
            (QuantumCircuit)

        """
        ...
    

class Thouless(Ansatz):
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

    def build_circuit(self, n, angles, num_electron_pairs, num_orbits):
        '''Quantum Circuits with n qubits and C(n 2)  R(theta)[p,q] gates
        
        Args:
            n(int): number of quantum qubits
            angles(List[float]): a list with C(N 2) parameter
            num_electron_pairs(int): number of electron pairs
            num_orbits(int): number of orbits

        Returns:
            Circuit: Quantum Circuits with N qubits and C(N 2) R gates
        '''
        # circuits with n qubits
        circuit = Circuit(n)

        gate = self.build_gate(n, angles, num_electron_pairs, num_orbits)

        gate | circuit
        
        return circuit

    def build_gate(self, n, angles, num_electron_pairs, num_orbits):
        """ 
        build a compositegate from a list of parameters

        Args:
            args(tuple): parameters of the ansatz builder
        
        Returns:
            (CompositeGate): a composite gate built from the given arguments

        """
        # circuits with n qubits
        ansatz = CompositeGate()

        R = self.modified_Givens_rotation

        with ansatz:
            # add gates to the circuits in parallelization
            if(n == 1):
                return ansatz
            elif(n == 2):
                R(angles.pop()) & [0, 1] # assign the parameters to each circuit
            elif(n > 2):
                i = 0

                while i < n - 1:
                    index = i
                    while index >= 0:
                        R(angles.pop()) & [n - index - 2, n - index - 1]
                        index -= 2
                    i += 1

                while i > 0:
                    index = i-1
                    while index >= 0:
                        R(angles.pop()) & [n - index - 2, n - index - 1]
                        index -= 2
                    i -= 1
            
        return ansatz


# test functions
if __name__ =='__main__':
    ansatz = Thouless.build_circuit(9, [i for i in range(40)])
    ansatz.draw()