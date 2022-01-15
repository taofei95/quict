#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/5 8:37
# @Author  : Han Yu
# @File    : Amplitude.py

from QuICT.algorithm import Algorithm
from QuICT import *
from QuICT.backends import systemCdll
from QuICT.simulation.CPU_simulator import CircuitSimulator
import numpy as np
from ctypes import c_int
from typing import List, Tuple


class Amplitude(Algorithm):
    """ get the amplitude of some circuit with some ancillary qubits which are ignored

    """

    @classmethod
    def run(cls, circuit: Circuit) -> Tuple[np.ndarray, List[int]]:
        """
        Args
        -----------
        circuit:
            circuit to be simulated

        Returns
        -------------
        A tuple with a complex numpy array representing the amplitude vector and measure results.
        """
        return CircuitSimulator(circuit.circuit_width()).run(circuit)
