# Copyright 2020 Google
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
An implementation of gradient based Restricted-Hartree-Fock

This uses a bunch of the infrastructure already used in the experiment
should only need an RHF_object.
"""
import numpy as np
import scipy as sp

def rhf_params_to_matrix(parameters: np.ndarray,
                         num_qubits: int,
                         occ: Optional[Union[None, List[int]]] = None,
                         virt: Optional[Union[None, List[int]]] = None):
    """Assemble variational parameters into a matrix.

    For restricted Hartree-Fock we have nocc * nvirt parameters. These are
    provided as a list that is ordered by (virtuals) \times (occupied) where
    occupied is a set of indices corresponding to the occupied orbitals w.r.t
    the Lowdin basis and virtuals is a set of indices of the virtual orbitals
    w.r.t the Lowdin basis.  For example, for H4 we have 2 orbitals occupied and
    2 virtuals:

    occupied = [0, 1]  virtuals = [2, 3]

    parameters = [(v_{0}, o_{0}), (v_{0}, o_{1}), (v_{1}, o_{0}),
                  (v_{1}, o_{1})]
               = [(2, 0), (2, 1), (3, 0), (3, 1)]

    You can think of the tuples of elements of the upper right triangle of the
    antihermitian matrix that specifies the c_{b, i} coefficients.

    coefficient matrix
    [[ c_{0, 0}, -c_{1, 0}, -c_{2, 0}, -c_{3, 0}],
     [ c_{1, 0},  c_{1, 1}, -c_{2, 1}, -c_{3, 1}],
     [ c_{2, 0},  c_{2, 1},  c_{2, 2}, -c_{3, 2}],
     [ c_{3, 0},  c_{3, 1},  c_{3, 2},  c_{3, 3}]]

    Since we are working with only non-redundant operators we know c_{i, i} = 0
    and any c_{i, j} where i and j are both in occupied or both in virtual = 0.
    """
    if occ is None:
        occ = range(num_qubits // 2)
    if virt is None:
        virt = range(num_qubits // 2, num_qubits)

    # check that parameters are a real array
    if not np.allclose(parameters.imag, 0):
        raise ValueError("parameters input must be real valued")

    kappa = np.zeros((len(occ) + len(virt), len(occ) + len(virt)))
    for idx, (v, o) in enumerate(product(virt, occ)):
        kappa[v, o] = parameters[idx].real
        kappa[o, v] = -parameters[idx].real
    return kappa

def rhf_minimization(rhf_object, method='CG', verbose=True):
    """Perform Hartree-Fock energy minimization.

    Args:
        rhf_object: RestrictedHartreeFockObject
        method: scipy.optimize.minimize method

    Returns:
        Scipy optimize result
    """

    initial_opdm = np.diag([1] * rhf_object.nocc +
                           [0] * rhf_object.nvirt)

    def unitary(params):
        kappa = rhf_params_to_matrix(
            params,
            len(rhf_object.occ) + len(rhf_object.virt), rhf_object.occ,
            rhf_object.virt)
        return sp.linalg.expm(kappa)
        
    def energy(params):
        u = unitary(params)
        final_opdm_aa = u @ initial_opdm @ np.conjugate(u).T
        tenergy = rhf_object.energy_from_opdm(final_opdm_aa)
        return tenergy

    def gradient(params):
        u = unitary(params)
        final_opdm_aa = u @ initial_opdm @ np.conjugate(u).T
        return rhf_object.global_gradient_opdm(params, final_opdm_aa).real


    init_guess = np.zeros(rhf_object.nocc * rhf_object.nvirt)

    return sp.optimize.minimize(energy,
                                init_guess,
                                jac=gradient,
                                method=method,
                                options={'disp': verbose})
