#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2022/4/26 8:48
# @Author  : Ou Shigang
# @File    : VQE.py

class VQE(object):
    """ implementation of simulated and quantum VQE"""

    def based_state_vectors(self, ansatz):
        """ calculate the measured result using state vectors"""
        ...

    def based_measure(self, ansatz):
        """ calculate the measured result using quantum circuits"""
        ...
    

class HFVQE(VQE):
    """ Hartree Fock method VQE implementation"""
    ...

