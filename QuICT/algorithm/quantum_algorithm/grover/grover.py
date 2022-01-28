#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/6/1 16:20 上午
# @Author  : Zhu Qinlin
# @File    : standard_grover.py

from QuICT.core.qubit.qubit import Qureg
import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.qcda.synthesis.mct import MCTLinearOneDirtyAux

from QuICT.simulation.gpu_simulator import ConstantStateVectorSimulator
import logging


def my_print(msg, demo_mode):
    if demo_mode:
        print(msg)
    else:
        logging.info(msg)


def degree_counterclockwise(v1: np.ndarray, v2: np.ndarray):
    """from v1 to v2
    """
    d = np.real(np.arccos(sum(v1*v2)/np.sqrt(sum(v1*v1)*sum(v2*v2))))
    if d > 0.5*np.pi:
        d = np.pi-d
    return d


class Grover:
    """ simple grover

    Quantum Computation and Quantum Information - Michael A. Nielsen & Isaac L. Chuang
    """
    @staticmethod
    def run(n, oracle, simulator=None, demo_mode=False, **kwargs):
        """ grover search for f with custom oracle

        Args:
            n(int): the length of input of f
            oracle(CompositeGate): the oracle assuming one ancilla@[n] in |->
        Returns:
            int: the a satisfies that f(a) = 1
        """
        if simulator is None:
            simulator = ConstantStateVectorSimulator()
        circuit = Circuit(n + 1)
        index_q:list = list(range(n))
        result_q:int = n
        N = 2**n
        theta = 2 * np.arccos(np.sqrt(1 - 1 / N))
        T = round(np.arccos(np.sqrt(1 / N)) / theta)
        my_print(
            f"[init] theta = {theta:.4f}, Grover iteration count = {T}", demo_mode)
        phase_size = 0
        oracle_size = 0

        # create equal superposition state in index_q
        for idx in index_q: H | circuit(idx)
        # create |-> in result_q
        X | circuit(result_q)
        H | circuit(result_q)
        for i in range(T):
            # Grover iteration
            if demo_mode and i==0:
                tmp = circuit.size()
            oracle | circuit
            if demo_mode and i==0:
                oracle_size = circuit.size() - tmp
                tmp = circuit.size()
            for idx in index_q: H | circuit(idx)
            # control phase shift
            for idx in index_q: X | circuit(idx)
            H | circuit(index_q[n - 1])

            # yet_gates = CompositeGate()
            # with yet_gates:
            #     yet_circuit = [j for j in range(0, n - 1)] + [index_q[n - 1]] + [result_q]
            #     yet_n = n + 1
            #     yet_controls = [i for i in range(yet_n - 2)]
            #     yet_target = yet_n - 2
            #     yet_aux = yet_n - 1      # this is a dirty ancilla
            #     one_dirty_aux(yet_gates, yet_controls, yet_target, yet_aux)
            # yet_gates | circuit
            MCTLinearOneDirtyAux.execute(n+1) | circuit

            H | circuit(index_q[n - 1])
            for idx in index_q: X | circuit(idx)
            # control phase shift end
            for idx in index_q: H | circuit(idx)
            if demo_mode and i==0:
                phase_size = circuit.circuit_size() - tmp
            if demo_mode:
                amp = simulator.run(circuit)
                amp = np.array(amp[::2])*np.sqrt(2)
                d = degree_counterclockwise(amp, kwargs["beta"])
                my_print(
                    f"[{i+1:3}-th Grover iteration] "
                    +f"degree from target state: {d:.3f} "
                    +f"success rate:{(np.real(amp[kwargs['target']])**2)*100:.1f}%", demo_mode)
        amp = simulator.run(circuit)
        if demo_mode:
            amp = np.array(amp[::2])*np.sqrt(2)
        for idx in index_q: Measure | circuit(idx)
        simulator.run(circuit)
        my_print(f"circuit width          = {circuit.width():4}", demo_mode)
        my_print(f"circuit depth          = {circuit.depth():4}", demo_mode)
        my_print(f"circuit size           = {circuit.size():4}", demo_mode)
        my_print(f"Grover iteration size  = {oracle_size:4}+{phase_size:4}", demo_mode)
        return int(circuit[index_q])
