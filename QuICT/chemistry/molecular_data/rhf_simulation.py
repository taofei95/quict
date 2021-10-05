from QuICT.chemistry.molecular_data.molecular_data import MolecularData
import os
import numpy as np
import scipy as sp

DATA_DIR = os.path.dirname(os.path.abspath(__file__)) + '/molecular_data'

def rhf_simulation(n_atoms, distance, data_dir=None):
    if data_dir == None:
        data_dir = DATA_DIR
    data_dir += "/hydrogen_chains"
    data_dir += "/h_" + n_atoms + "_sto-3g"
    data_dir += "/bond_distance_" + distance
    molfile = data_dir + "/H" + n_atoms
    molfile += "_sto-3g_singlet_linear_r-" + distance
    moldata = MolecularData(molfile)

    S = np.load(data_dir + "/overlap.npy")
    Hcore = np.load(data_dir + "/h_core.npy")
    TEI = np.load(data_dir + "/tei.npy")

    _, X = sp.linalg.eigh(Hcore, S)
    

    return moldata


    
