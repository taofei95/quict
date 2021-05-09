from enum import Enum
from collections import deque
from copy import copy

import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


from .utility import *

from QuICT.tools.interface import *
from QuICT.core.circuit import *
from QuICT.core.gate import *
from QuICT.core.gate.gate import *
from QuICT.core.exception import *
from QuICT.core.layout import *


class RandomCircuitGenerator(object):
    def __init__(self, minimum: int = 50, maximum: int = 2000, min_num_of_qubits: int = 5, max_num_of_qubits: int = 20, seed: int = 0):
        self._minimum = minimum
        self._maximum = maximum
        self._min_num_of_qubits = min_num_of_qubits
        self._max_num_of_qubits = max_num_of_qubits
        self._prg = np.random.default_rng(seed * int(time.time()))

    def __call__(self)->Circuit:
        size = self._prg.integers(low = self._minimum, high = self._maximum)
        num_of_qubits = self._prg.integers(low = self._min_num_of_qubits, high = self._max_num_of_qubits)
        return self._generate_random_circuit(circuit_size = size, num_of_qubits = num_of_qubits)

    def run(self, file_path: str):
        pass

    
    def _generate_random_circuit(self, circuit_size: int = 0, num_of_qubits: int = 20)->Circuit:
        pre_gate_qubits = [-1,-1]
        res_circuit = Circuit(wires = num_of_qubits)
        for _ in range(circuit_size):
            qubits = self._generate_two_qubit_gate(pre_gate_qubits, num_of_qubits)
            GateBuilder.setGateType(GATE_ID['cx']) 
            GateBuilder.setCargs(int(qubits[0]))
            GateBuilder.setTargs(int(qubits[1]))
            res_circuit.append(GateBuilder.getGate())
            pre_gate_qubits = qubits
        return res_circuit

    def _generate_two_qubit_gate(self, pre_gate_qubits: Tuple[int, int], num_of_qubits: int)->Tuple[int, int]:
        res_qubits = self._prg.integers(low = 0, high = num_of_qubits, size = 2)
        while self._is_valid_qubits(res_qubits, pre_gate_qubits) is not True:
            res_qubits = self._prg.integers(low = 0, high = num_of_qubits, size = 2)
        return res_qubits


    def _is_valid_qubits(self, qubits: Tuple[int, int], pre_gate_qubits: Tuple[int, int])->bool:
        if qubits[0] == qubits[1]:
            return False
 
        if (not len(pre_gate_qubits)==0) and (qubits[0] == pre_gate_qubits[0] and qubits[1] == pre_gate_qubits[1]) or (qubits[1] == pre_gate_qubits[0] and qubits[0] == pre_gate_qubits[1]):
            return False
        return True 
