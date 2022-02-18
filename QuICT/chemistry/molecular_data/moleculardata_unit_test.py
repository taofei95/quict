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
    assert moldata.n_orbitals == 6
    assert moldata.n_electrons == 6
    assert moldata.nuclear_repulsion == 3.5414167175607694
    assert moldata.one_body_integrals == None
    assert moldata.two_body_integrals == None

    data_dir = os.path.dirname(__file__) + "/molecular_data/hydrogen_chains/h_6_sto-3g/bond_distance_1.7"
    moldata = MolecularData(data_dir + "/H6_sto-3g_singlet_linear_r-1.7")
    assert moldata.n_orbitals == 6
    assert moldata.n_electrons == 6
    assert moldata.nuclear_repulsion == 2.7081421957817655
    assert moldata.one_body_integrals == None
    assert moldata.two_body_integrals == None

if __name__ == "__main__":
    pytest.main(["./moleculardata_unit_test.py"])