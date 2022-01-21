#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/9/11 18:20
# @Author  : Xiaoquan Xu
# @File    : rhf_simulation.py

import os
import numpy as np
import scipy as sp
from QuICT.chemistry.molecular_data import molecular_data
from QuICT.chemistry.molecular_data import RHFObjective

from QuICT.chemistry.molecular_data.hamiltonian import *
from QuICT.chemistry.molecular_data.molecular_data import MolecularData

DATA_DIR = os.path.dirname(os.path.abspath(__file__)) + '/molecular_data'

def rhf_simulation(n_atoms, distance, data_dir=None):
    if data_dir == None:
        data_dir = DATA_DIR
    data_dir += "/hydrogen_chains"
    data_dir += "/h_" + str(n_atoms) + "_sto-3g"
    data_dir += "/bond_distance_" + str(distance)
    molfile = data_dir + "/H" + str(n_atoms)
    molfile += "_sto-3g_singlet_linear_r-" + str(distance)
    moldata = MolecularData(molfile)

    S = np.load(data_dir + "/overlap.npy")
    Hcore = np.load(data_dir + "/h_core.npy")
    TEI = np.einsum("ikjl", np.load(data_dir + "/tei.npy"))

    _, X = sp.linalg.eigh(Hcore, S)
    obi = obi_basis_rotation(Hcore, X)
    tbi = tbi_basis_rotation(TEI, X)
    molecular_hamiltonian = generate_hamiltonian(moldata.nuclear_repulsion, obi, tbi)

    rhf_objective = RHFObjective(molecular_hamiltonian, moldata.n_electrons)
    result = rhf_objective.minimization() 
    
    return rhf_objective, moldata, result, obi, tbi