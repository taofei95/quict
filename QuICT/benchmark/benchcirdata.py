import numpy as np
import re
import math

from QuICT.core import Circuit
from QuICT.core.utils.gate_type import GateType
from QuICT.qcda.utility.circuit_cost import CircuitCost


class BenchCirData:
    """ A data structure for storing benchmark information. """
    @property
    def circuit(self) -> Circuit:
        """ Return the circuits from QuantumMachinebenchmark. """
        return self._circuit

    @property
    def machine_amp(self) -> list:
        """ Return the quantum machine sample result of circuits. """
        return self._machine_amp

    @machine_amp.setter
    def machine_amp(self, machine_amp: list):
        assert isinstance(machine_amp, list)
        self._machine_amp = machine_amp

    @property
    def type(self) -> str:
        """ Return the field of circuits. """
        return self._type

    @property
    def field(self) -> str:
        """ Return the field of circuits. """
        return self._field

    @property
    def level(self) -> list:
        """ Return the level of circuits. """
        return self._level

    @property
    def width(self) -> int:
        """ Return the qubit number of circuit. """
        return self._width

    @property
    def size(self) -> int:
        """ Return the gate number of circuit. """
        return self._size

    @property
    def depth(self) -> int:
        """ Return the depth of circuit. """
        return self._depth
    
    @property
    def benchmark_score(self) -> float:
        """ Return the general benchmark score of each circuit. """
        self._evaluate_circuits_score()
        return self._benchmark_score

    @property
    def I_score(self) -> float:
        """ Return the inital score of quantum machine. """
        return self._I_score
    
    @property
    def D_score(self) -> float:
        """ Return the difficulty score of quantum machine. """
        return self._D_score
    
    @property
    def E_score(self) -> float:
        """ Return the execution score of quantum machine. """
        return self._E_score

    @property
    def qv(self) -> int:
        """ Return the quantum volume of circuit. """
        return self._qv

    def _level_score(self) -> float:
        """ Return the level index of circuit. """
        level_score = round(self._level / (self._level + 1), 3)
        return level_score

    def _circuit_cost(self) -> float:
        """ Return the cost of algorithmic circuit. """
        cc = CircuitCost()
        circuit_cost = cc.evaluate_cost(self.circuit)
        return circuit_cost

    def _bench_metric(self) -> float:
        """ Return the index of special benchmark circuit. """
        cir_property = self._circuit.name
        bench_metric = float(cir_property.split("+")[-1])

        return bench_metric

    def _calculate_entropy(self, p, q):
        def normalization(data):
            data = np.array(data)
            data = data / np.sum(data)
            return data

        sum = 0.0
        delta = 1e-7
        p = abs(normalization(p))
        q = abs(normalization(q))
        for x in map(lambda y, p: (1 - y) * math.log(1 - p + delta) + y * math.log(p + delta), p, q):
            sum += x
        cross_entropy = 1 - (-sum / len(p))

        return cross_entropy

    def _calculate_fidelity(self):
        if self._type != "algorithm":
            self._fidelity = self._machine_amp[0]
        else:
            width = self.width
            if self._field == "adder":
                self._fidelity = self._machine_amp[0]
            elif self._field == "qft":
                p, q = self._machine_amp, [float(1 / (2 ** int(width)))] * (2 ** int(width))
                self._fidelity = self._calculate_entropy(p, q)
            elif self._field == "cnf":
                self._fidelity = self._machine_amp[8]
            elif self._field == "quantum_walk":
                self._fidelity = self._machine_amp[3] + self._machine_amp[-2]
            elif self._field == "vqe":
                index = '1' * int(width / 2)
                if width % 2 == 1:
                    index += '0' * (int(width / 2) + 1)
                else:
                    index += '0' * int(width / 2)
                self._fidelity = self._machine_amp[int(index, 2)]

    def _linear_mapping(self, score, s1, s2, t1, t2):
        # The range of input fractions is s1 to s2
        # The output is mapped to the interval t1 to t2

        # Calculate the mapping ratio
        scale = (t2 - t1) / (s2 - s1)
        # Calculate the mapping result
        mapped_score = round((score - s1) * scale + t1, 3)
        return mapped_score

    def _graded_mapping(self, value):
        value = math.log(value, 2)
        if value >= 1 and value <= 10:
            mapped_value = self._linear_mapping(value, 1, 10, 50, 60)
        elif value > 10 and value <= 50:
            mapped_value = self._linear_mapping(value, 10, 50, 60, 70)
        else:
            mapped_value = 80
        return mapped_value

    def _evaluate_circuits_score(self):
        # initial score
        self._I_score = self._graded_mapping(self._qv)

        # Difficulty Score
        if self.type == "random":
            level = self._level_score()
            self._D_score = self._linear_mapping(level, 0, 1, 1, 2)
        elif self.type == "algorithm":
            cost = self._circuit_cost()
            cost_max = self.width * 100
            self._D_score = self._linear_mapping(cost, 0, cost_max, 1, 2)
        elif self.type == "benchmark":
            metric = self._bench_metric()
            self._D_score = self._linear_mapping(metric, 0, 1, 1, 2)

        # Execution Score
        self._calculate_fidelity()
        fidelity = round(self._fidelity, 4)
        value = 2 / 3
        if fidelity >= value:
            self._E_score = self._linear_mapping(fidelity, 0, 1, 1.5, 2)
        elif fidelity >= 1 - value and fidelity < value:
            self._E_score = self._linear_mapping(fidelity, 0, 1, 1, 1.5)
        elif fidelity > 0 and fidelity < 1 - value:
            self._E_score = self._linear_mapping(fidelity, 0, 1, 0.5, 1)
        elif fidelity == 0:
            self._E_score = 0

        # Total Score
        cir_score = round(self._I_score * self._D_score * self._E_score)
        self._benchmark_score = cir_score

    def __init__(
        self,
        circuit: Circuit
    ):
        """
        Args:
            circuits (list, optional): The list of circuit which from QuantumMachinebenchmark.
        """
        assert isinstance(circuit, Circuit)
        self._circuit = circuit
        self._machine_amp = []

        # Score related
        self._fidelity = 0
        self._I_score = 0
        self._D_score = 0
        self._E_score = 0
        self._benchmark_score = 0

        # Circuit related
        cir_property = self._circuit.name
        circuit_info = re.findall(r"\d+", cir_property)
        self._level = int(cir_property.split("+")[3][-1])
        self._type = cir_property.split("+")[:-1][0]
        self._field = cir_property.split("+")[:-1][1]
        self._width = int(circuit_info[0])
        self._size = int(circuit_info[1])
        self._depth = int(circuit_info[2])

        self._qv = 2 ** min(self._width, self._depth)
