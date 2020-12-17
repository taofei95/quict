#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/23 4:31
# @Author  : Han Yu
# @File    : qubit_unit_test.py

from QuICT.core import Circuit, X, H, Measure

def test_Qubit_Attributes_prob():
    circuit = Circuit(3)
    X       | circuit(0)
    Measure | circuit(0)
    circuit.exec()
    if circuit[0].measured != 1:
        assert 0
    if abs(circuit[0].prob - 1) > 1e-10:
        assert 0
    H       | circuit(1)
    Measure | circuit(1)
    circuit.exec()
    if abs(circuit[1].prob - 0.5) > 1e-10:
        print(circuit[1].prob)
        assert 0

def test_Qureg_Function_slice():
    circuit = Circuit(10)
    qureg = circuit.qubits
    slices = qureg[2:7]
    i = qureg[0].id + 2
    for qubit in slices:
        if qubit.id != i:
            print(qubit.id, i)
            assert 0
        i = i + 1


if __name__ == "__main__":
    # pytest.main(["./_unit_test.py", "./circuit_unit_test.py", "./gate_unit_test.py", "./qubit_unit_test.py"])
    pytest.main(["./qubit_unit_test.py"])
