#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/8/22 22:06
# @Author  : Xiaoquan Xu
# @File    : encoder_unit_test.py

import pytest

from QuICT.chemistry.operator.encoder import Encoder, JordanWigner, Parity, BravyiKitaev
from QuICT.chemistry.operator.fermion_operator import FermionOperator
from QuICT.chemistry.operator.qubit_operator import QubitOperator

anni_0 = FermionOperator('0')
crea_0 = FermionOperator('0^')
anni_2 = FermionOperator('2')
crea_2 = FermionOperator('2^')
anni_5 = FermionOperator('5')
crea_5 = FermionOperator('5^')
    
def test_JordanWigner():
    JW = JordanWigner(7)
    assert JW.encoder(anni_0) == QubitOperator("X0", 0.5) + QubitOperator("Y0", 0.5j)
    assert JW.encoder(crea_0) == QubitOperator("X0", 0.5) - QubitOperator("Y0", 0.5j)
    assert JW.encoder(anni_2) == QubitOperator("Z0 Z1 X2", 0.5) + QubitOperator("Z0 Z1 Y2", 0.5j)
    assert JW.encoder(crea_2) == QubitOperator("Z0 Z1 X2", 0.5) - QubitOperator("Z0 Z1 Y2", 0.5j)
    assert JW.encoder(anni_5) == QubitOperator("Z0 Z1 Z2 Z3 Z4 X5", 0.5) \
        + QubitOperator("Z0 Z1 Z2 Z3 Z4 Y5", 0.5j)
    assert JW.encoder(crea_5) == QubitOperator("Z0 Z1 Z2 Z3 Z4 X5", 0.5) \
        - QubitOperator("Z0 Z1 Z2 Z3 Z4 Y5", 0.5j)
    assert JW.encoder(crea_2 * anni_2) == QubitOperator([], 0.5) - QubitOperator("Z2", 0.5)
    assert JW.encoder(crea_5 * anni_5) == QubitOperator([], 0.5) - QubitOperator("Z5", 0.5)

def test_Parity():
    PR = Parity(6)
    assert PR.encoder(anni_2) == QubitOperator("Z1 X2 X3 X4 X5", 0.5) + QubitOperator("Y2 X3 X4 X5", 0.5j)
    assert PR.encoder(crea_2) == QubitOperator("Z1 X2 X3 X4 X5", 0.5) - QubitOperator("Y2 X3 X4 X5", 0.5j)
    assert PR.encoder(anni_5) == QubitOperator("Z4 X5", 0.5) + QubitOperator("Y5", 0.5j)
    assert PR.encoder(crea_5) == QubitOperator("Z4 X5", 0.5) - QubitOperator("Y5", 0.5j)
    assert PR.encoder(crea_2 * anni_2) == QubitOperator([], 0.5) - QubitOperator("Z1 Z2", 0.5)
    assert PR.encoder(crea_5 * anni_5) == QubitOperator([], 0.5) - QubitOperator("Z4 Z5", 0.5)

    assert PR.encoder(crea_5 + anni_5) == QubitOperator("Z4 X5")
    assert PR.encoder(crea_5 + anni_5, 8) == QubitOperator("Z4 X5 X6 X7")

def test_BravyiKitaev():
    BK = BravyiKitaev() # default = 10
    assert BK.encoder(crea_0 + anni_0) == QubitOperator("X0 X1 X3 X7")
    assert BK.encoder(crea_2 - anni_2) == QubitOperator("Z1 Y2 X3 X7", -1j)
    assert BK.encoder(crea_5 + anni_5) == QubitOperator("Z3 Z4 X5 X7")

    assert BK.encoder(FermionOperator('0')+FermionOperator('0^'), 50) \
        == QubitOperator("X0 X1 X3 X7 X15 X31") 
    assert BK.encoder(FermionOperator('50')+FermionOperator('50^'), 100) \
        == QubitOperator("Z31 Z47 Z49 X50 X51 X55 X63")
    assert BK.encoder(FermionOperator('62')-FermionOperator('62^'), 100) \
        == QubitOperator("Z31 Z47 Z55 Z59 Z61 Y62 X63", 1j)

    assert BK.encoder(anni_2) == QubitOperator("Z1 X2 X3 X7", 0.5) + QubitOperator("Z1 Y2 X3 X7", 0.5j)
    assert BK.encoder(crea_2) == QubitOperator("Z1 X2 X3 X7", 0.5) - QubitOperator("Z1 Y2 X3 X7", 0.5j)
    assert BK.encoder(anni_5) == QubitOperator("Z3 Z4 X5 X7", 0.5) + QubitOperator("Z3 Y5 X7", 0.5j)
    assert BK.encoder(crea_5) == QubitOperator("Z3 Z4 X5 X7", 0.5) - QubitOperator("Z3 Y5 X7", 0.5j)

    assert BK.encoder(crea_2 * anni_2) == QubitOperator([], 0.5) - QubitOperator("Z2", 0.5)
    assert BK.encoder(crea_5 * anni_5) == QubitOperator([], 0.5) - QubitOperator("Z4 Z5", 0.5)

if __name__ == "__main__":
    pytest.main(["./encoder_unit_test.py"])