#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/12 11:49
# @Author  : Han Yu
# @File    : unit_test.py

import pytest
from QuICT.core import *
from QuICT.qcda.synthesis.mct import MCTOneAux, MCTLinearSimulationHalfDirtyAux, MCTLinearSimulationOneDirtyAux
from QuICT.algorithm import SyntheticalUnitary

def Set(qreg, N):
    """
    Set the qreg as N, using X gates on specific qubits
    """
    n = len(qreg)
    for i in range(n):
        if N % 2 == 1:
            X | qreg[n - 1 - i]
        N = N // 2

def test_MCT_Linear_Simulation_Half():
    max_qubit = 11
    for i in range(3, max_qubit):
        for m in range(1, i // 2 + (1 if i % 2 == 1 else 0)):
            circuit = Circuit(i)
            MCTLinearSimulationHalfDirtyAux.execute(m, i) | circuit
            unitary = SyntheticalUnitary.run(circuit, showSU=False)
            for j in range(1 << i):
                for k in range(1 << i):
                    flag = True
                    for l in range(i):
                        jj = j & (1 << (i - l - 1))
                        kk = k & (1 << (i - l - 1))
                        if l < m:
                            if jj == 0 or kk == 0:
                                flag = False
                            if jj != kk:
                                if abs(abs(unitary[j, k])) > 1e-30:
                                    print(i, j, k, unitary[j, k])
                                    assert 0
                                break
                        elif l < i - 1:
                            if jj != kk:
                                if abs(abs(unitary[j, k])) > 1e-30:
                                    print(i, j, k, unitary[j, k])
                                    assert 0
                                break
                        else:
                            if flag:
                                if jj == kk:
                                    if abs(abs(unitary[j, k])) > 1e-30:
                                        print(i, m, j, k, unitary[j, k])
                                        print(unitary)
                                        circuit.print_information()
                                        assert 0
                                else:
                                    if abs(abs(unitary[j, k] - 1)) > 1e-30:
                                        print(i, j, k, unitary[j, k])
                                        circuit.print_information()
                                        assert 0
                            else:
                                if jj != kk:
                                    if abs(abs(unitary[j, k])) > 1e-30:
                                        print(i, j, k, unitary[j, k])
                                        assert 0
                                else:
                                    if abs(abs(unitary[j, k] - 1)) > 1e-30:
                                        print(i, m, j, k, unitary[j, k])
                                        circuit.print_information()
                                        print(unitary)
                                        assert 0
    assert 1

    
def test_MCT_Linear_Simulation_One():
    max_control_bit = 10
    for n in range(1, max_control_bit):
        for control_bits in range(0,2**n):
            circuit = Circuit(n + 2)
            controls = circuit([i for i in range(n)])
            aux = circuit(n)
            target = circuit(n + 1)
            Set(controls, control_bits)
            H | aux
            print("%d bits control = %d" %(n,control_bits))
            MCTLinearSimulationOneDirtyAux.execute(n) | (controls, aux, target)
            Measure | circuit
            circuit.exec()
            if (control_bits == 2**n - 1 and int(target) == 0) or (control_bits != 2**n - 1 and int(target) == 1):
                print("when control bits are %d, the targe is %d" %(control_bits, int(target)))
                assert 0
    
    assert 1


def test_MCT():
    max_qubit = 11
    for i in range(3, max_qubit):
        circuit = Circuit(i)
        MCTOneAux.execute(i) | circuit
        # assert 0
        unitary = SyntheticalUnitary.run(circuit)
        circuit.print_information()
        for j in range(1 << i):
            flagj = True
            for l in range(2, i):
                if (j & (1 << l)) == 0:
                    flagj = False
                    break
            for k in range(1 << i):
                flag = flagj
                for l in range(2, i):
                    if (k & (1 << l)) == 0:
                        flag = False
                        break
                if flag:
                    if ((k & 1) != (j & 1)) or ((k & 2) == (j & 2)):
                        if abs(abs(unitary[j, k])) > 1e-10:
                            print(i, j, k, unitary[j, k])
                            assert 0
                    else:
                        if abs(abs(unitary[j, k] - 1)) > 1e-10:
                            print(i, j, k, unitary[j, k])
                            assert 0
                else:
                    if j == k:
                        if abs(abs(unitary[j, k] - 1)) > 1e-10:
                            print(i, j, k, unitary[j, k])
                            print(range(2, i), 1 << 2, j & (1 << 2), k & (1 << 2))
                            print(unitary)
                            assert 0
                    else:
                        if abs(abs(unitary[j, k])) > 1e-10:
                            print(i, j, k, unitary[j, k])
                            assert 0

    assert 1

if __name__ == "__main__":
    pytest.main(["./unit_test.py"])
