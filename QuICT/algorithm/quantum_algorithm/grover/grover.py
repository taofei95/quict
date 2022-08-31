#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/6/1 16:20 上午
# @Author  : Zhu Qinlin
# @File    : standard_grover.py

import numpy as np
import logging

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.simulation.cpu_simulator import CircuitSimulator


class Grover:
    """ The Grover Algorithm.

        Quantum Computation and Quantum Information - Michael A. Nielsen & Isaac L. Chuang
    """

    def __init__(self, simulator=CircuitSimulator()):
        """ Initialize the simulator circuit of Grover algorithm.

        Args:
            simulator (Union[ConstantStateVectorSimulator, CircuitSimulator], optional):
                The simulator for simulating quantum circuit. Defaults to CircuitSimulator().
        """
        self._simulator = simulator

    def run(self, index_qubits: int, oracle_qubits: int, oracle):
        """ Grover search for an index register with costumed oracle.

        Args:
            index_qubits (int): The number of qubits of the index register.
            oracle_qubits (int): The number of qubits of the oracle working space.
            oracle (CompositeGate): The costumed oracle which aims to flip phase of the target state.

        Returns:
            int: The target index to be searched.
        """

        assert oracle_qubits > 0, "The oracle should contain at least 1 qubit, which is shared by the MCT part."

        circuit = Circuit(index_qubits + oracle_qubits)
        index_q = list(range(index_qubits))
        oracle_q = list(range(index_qubits, index_qubits + oracle_qubits))
        N = 2 ** index_qubits
        theta = 2 * np.arccos(np.sqrt(1 - 1 / N))
        steps = round(np.arccos(np.sqrt(1 / N)) / theta)

        # create equal superposition state in index_q
        for idx in index_q:
            H | circuit(idx)
        # rotation
        for i in range(steps):
            # Grover iteration
            oracle | circuit(index_q + oracle_q)
            # s_c = np.ones((1 << index_qubits, 1 << index_qubits)) / (1 << index_qubits)
            # matrix = -np.eye(1 << index_qubits) + 2 * s_c
            # Unitary(matrix) | circuit(index_q)
            for idx in index_q:
                H | circuit(idx)
            # control phase shift
            shift = -1 * np.identity(2 ** index_qubits)
            shift[0, 0] = 1
            Unitary(shift) | circuit(index_q)
            # control phase shift end
            for idx in index_q:
                H | circuit(idx)
        for idx in index_q:
            Measure | circuit(idx)

        self._simulator.run(circuit)
        logging.info(f"circuit width          = {circuit.width():4}")
        logging.info(f"circuit depth          = {circuit.depth():4}")
        logging.info(f"circuit size           = {circuit.size():4}")
        return int(circuit[index_q])
