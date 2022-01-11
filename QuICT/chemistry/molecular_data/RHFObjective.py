#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2022/1/11 15:36
# @Author  : Xiaoquan Xu
# @File    : RHFObjective.py


from QuICT.chemistry.molecular_data.hamiltonian import Hamiltonian


class RHFObjective:
    def __init__(self, hamiltonian, num_electrons: int):
        self.hamiltonian = hamiltonian
        self.fermion_hamiltonian = hamiltonian.get_fermion_operator()
        self.num_qubits = hamiltonian.one_body_tensor.shape[0]
        self.num_orbitals = self.num_qubits // 2
        self.num_electrons = num_electrons
        self.nocc = self.num_electrons // 2
        self.nvirt = self.num_orbitals - self.nocc
        self.occ = list(range(self.nocc))
        self.virt = list(range(self.nocc, self.nocc + self.nvirt))
    
    def minimization():
        pass

