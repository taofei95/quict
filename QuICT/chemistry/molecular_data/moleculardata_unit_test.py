#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/9/14 15:51
# @Author  : Xiaoquan Xu
# @File    : moleculardata_unit_test.py

import os
import pytest
from QuICT.chemistry.molecular_data.moleculardata import MolecularData

def test_load():
    data_dir = os.path.dirname(__file__) + "/molecular_data/hydrogen_chains/h_6_sto-3g/bond_distance_1.3"
    moldata = MolecularData(data_dir + "/H6_sto-3g_singlet_linear_r-1.3")
    print(moldata.n_orbitals)
    print(moldata.n_electrons)
    print(moldata.nuclear_repulsion)
    print(moldata.one_body_integrals)
    print(moldata.two_body_integrals)
    assert moldata.n_orbitals == 0

def test_save_and_load():
    pass

if __name__ == "__main__":
    pytest.main(["./moleculardata_unit_test.py"])