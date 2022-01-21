#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2022/1/11 15:36
# @Author  : Xiaoquan Xu
# @File    : RHFObjective.py

import numpy as np
import scipy as sp
import itertools as it
from QuICT.chemistry.molecular_data.hamiltonian import Hamiltonian

class RHFObjective:
    def __init__(self, hamiltonian: Hamiltonian, num_electrons: int):
        self.hamiltonian = hamiltonian
        self.fermion_hamiltonian = hamiltonian.get_fermion_operator()
        self.num_qubits = hamiltonian.obi.shape[0]
        self.num_orbitals = self.num_qubits // 2
        self.num_electrons = num_electrons
        self.nocc = self.num_electrons // 2
        self.nvirt = self.num_orbitals - self.nocc
        self.occ = list(range(self.nocc))
        self.virt = list(range(self.nocc, self.num_orbitals))

    def energy_from_opdm(self, opdm_aa: np.ndarray) -> float:
        """Return the energy.

        Args:
            opdm: The alpha-alpha block of the RDM
        """
        opdm = np.zeros((self.num_qubits, self.num_qubits), dtype=complex)
        opdm[::2, ::2] = opdm_aa
        opdm[1::2, 1::2] = opdm_aa
        tpdm = generate_tpdm(opdm)
        rdms = Hamiltonian(1, opdm, tpdm)
        return rdms.expectation(self.hamiltonian).real

    def global_gradient_opdm(self, params: np.ndarray, alpha_opdm: np.ndarray):
        opdm = np.zeros((self.num_qubits, self.num_qubits), dtype=np.complex128)
        opdm[::2, ::2] = alpha_opdm
        opdm[1::2, 1::2] = alpha_opdm
        tpdm = generate_tpdm(opdm)

        # now go through and generate all the necessary Z, Y, Y_kl matrices
        kappa_matrix = rhf_params_to_matrix(params, self.occ, self.virt)
        kappa_matrix_full = np.kron(kappa_matrix, np.eye(2))
        w_full, v_full = np.linalg.eigh(
            -1j * kappa_matrix_full)  # so that kappa = i U lambda U^
        eigs_scaled_full = get_matrix_of_eigs(w_full)

        grad = np.zeros(self.nocc * self.nvirt, dtype=np.complex128)
        kdelta = np.eye(self.num_qubits)

        # NOW GENERATE ALL TERMS ASSOCIATED WITH THE GRADIENT!!!!!!
        for p in range(self.nocc * self.nvirt):
            grad_params = np.zeros_like(params)
            grad_params[p] = 1
            Y = rhf_params_to_matrix(grad_params, self.occ, self.virt)
            Y_full = np.kron(Y, np.eye(2))

            # Now rotate Y into the basis that diagonalizes Z
            Y_kl_full = v_full.conj().T @ Y_full @ v_full
            # now rotate
            # Y_{kl} * (exp(i(l_{k} - l_{l})) - 1) / (i(l_{k} - l_{l}))
            # into the original basis
            pre_matrix_full = v_full @ (eigs_scaled_full *
                                        Y_kl_full) @ v_full.conj().T

            grad_expectation = -1.0 * np.einsum(
                'ab,pq,aq,pb',
                self.hamiltonian.obi,
                pre_matrix_full,
                kdelta,
                opdm,
                optimize='optimal').real
            grad_expectation += 1.0 * np.einsum(
                'ab,pq,bp,aq',
                self.hamiltonian.obi,
                pre_matrix_full,
                kdelta,
                opdm,
                optimize='optimal').real
            grad_expectation += 1.0 * np.einsum(
                'ijkl,pq,iq,jpkl',
                self.hamiltonian.tbi,
                pre_matrix_full,
                kdelta,
                tpdm,
                optimize='optimal').real
            grad_expectation += -1.0 * np.einsum(
                'ijkl,pq,jq,ipkl',
                self.hamiltonian.tbi,
                pre_matrix_full,
                kdelta,
                tpdm,
                optimize='optimal').real
            grad_expectation += -1.0 * np.einsum(
                'ijkl,pq,kp,ijlq',
                self.hamiltonian.tbi,
                pre_matrix_full,
                kdelta,
                tpdm,
                optimize='optimal').real
            grad_expectation += 1.0 * np.einsum(
                'ijkl,pq,lp,ijkq',
                self.hamiltonian.tbi,
                pre_matrix_full,
                kdelta,
                tpdm,
                optimize='optimal').real
            grad[p] = grad_expectation

        return grad

    def minimization(self, method='CG', verbose=True):
        """
        Perform Hartree-Fock energy minimization.

        Args:
            method: scipy.optimize.minimize method

        Returns:
            Scipy optimize result
        """

        state=[]
        for _ in range(self.num_orbitals):
            if _ < self.nocc:
                state += [1]
            else:
                state += [0]
        initial_opdm = np.diag(state)

        def unitary(params):
            kappa = rhf_params_to_matrix(params, self.occ, self.virt)
            return sp.linalg.expm(kappa)
            
        def energy(params):
            u = unitary(params)
            final_opdm_aa = u @ initial_opdm @ np.conjugate(u).T
            tenergy = self.energy_from_opdm(final_opdm_aa)
            return tenergy

        def gradient(params):
            u = unitary(params)
            final_opdm_aa = u @ initial_opdm @ np.conjugate(u).T
            return self.global_gradient_opdm(params, final_opdm_aa).real


        init_guess = np.zeros(self.nocc * self.nvirt)

        return sp.optimize.minimize(energy,
                                    init_guess,
                                    jac=gradient,
                                    method=method,
                                    options={'disp': verbose})

def get_matrix_of_eigs(w: np.ndarray) -> np.ndarray:
    r"""Transform the eigenvalues for getting the gradient.

    .. math:
        f(w) \rightarrow
        \frac{e^{i (\lambda_{i} - \lambda_{j})}}{i (\lambda_{i} - \lambda_{j})}

    Args:
        w: eigenvalues of C-matrix

    Returns:
        New array of transformed eigenvalues
    """
    transform_eigs = np.zeros((w.shape[0], w.shape[0]), dtype=np.complex128)
    for i, j in it.product(range(w.shape[0]), repeat=2):
        if np.isclose(abs(w[i] - w[j]), 0):
            transform_eigs[i, j] = 1
        else:
            transform_eigs[i, j] = (np.exp(1j *
                                           (w[i] - w[j])) - 1) / (1j *
                                                                  (w[i] - w[j]))
    return transform_eigs

def rhf_params_to_matrix(parameters, occ, virt):
    """Assemble variational parameters into a anti-Hermitian matrix.

    Args:
        parameters: list of length n_virt * n_occ

    Returns:
        A matrix kappa of size (n_occ + n_virt) * (n_occ + n_virt)
    
    If i and j both in occ or virt, K[i][j]=0
    If i in virt and j in occ, K[i][j]=parameters[]
    If i in occ and j in virt, K[i][j]=-parameters[]
    """

    # parameters must be real
    for p in parameters:
        if p.imag != 0:
            raise ValueError("parameters input must be real valued")

    kappa = np.zeros((len(occ) + len(virt), len(occ) + len(virt)))
    for idx, (v, o) in enumerate(it.product(virt, occ)):
        kappa[v, o] = parameters[idx].real
        kappa[o, v] = -parameters[idx].real
    return kappa

def generate_tpdm(opdm):
    n=opdm.shape[0]
    tpdm = np.zeros((n, n, n, n), dtype=complex)
    for p in range(n):
        for q in range(n):
            for r in range(n):
                for s in range(n):
                    tpdm[p,q,r,s]=opdm[p,r]-opdm[q,s]
    return tpdm