import numpy as np
import re
import math

from QuICT.core import Circuit


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
    def benchmark_score(self) -> float:
        """ Return the general benchmark score of each circuit. """
        return self._benchmark_score

    @benchmark_score.setter
    def benchmark_score(self, benchmark_score: float):
        self._benchmark_score = benchmark_score

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
    def qv(self) -> int:
        """ Return the quantum volume of circuit. """
        cir_attribute = re.findall(r"\d+", self.circuit.name)
        self._qv = (2 ** min(int(cir_attribute[0]), int(cir_attribute[2])))
        return self._qv

    @property
    def level_score(self) -> float:
        """ Return the quantum volume of circuit. """
        return self._level_score

    @property
    def fidelity(self) -> float:
        """ Return the fidelity of circuit. """
        if self._fidelity == 0:
            self._calculate_fidelity()
        self._fidelity = round(self._fidelity, 4)
        return self._fidelity

    def _calculate_fidelity(self):
        def calculate_entropy(p, q):
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

        split_list = list(np.linspace(0.0, 1.0, num=6)[3:])
        if self._type != "algorithm":
            self._fidelity = self._machine_amp[0]
            self._level_score = split_list[self.level - 1]
        else:
            width = self.width
            if self._field == "adder":
                self._fidelity = self._machine_amp[0]
                self._level_score = split_list[1]
            elif self._field == "qft":
                p, q = self._machine_amp, [float(1 / (2 ** int(width)))] * (2 ** int(width))
                self._fidelity = calculate_entropy(p, q)
                self._level_score = split_list[1]
            elif self._field == "cnf":
                self._fidelity = self._machine_amp[8]
                self._level_score = split_list[0]
            elif self._field == "quantum_walk":
                self._fidelity = self._machine_amp[3] + self._machine_amp[-2]
                self._level_score = split_list[2]
            elif self._field == "vqe":
                index = '1' * int(width / 2)
                if width % 2 == 1:
                    index += '0' * (int(width / 2) + 1)
                else:
                    index += '0' * int(width / 2)
                self._fidelity = self._machine_amp[int(index, 2)]
                self._level_score = split_list[0]

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
        self._benchmark_score = 0

        # Score related
        self._qv = 0
        self._bench_cir_value = 0
        self._fidelity = 0
        self._level_score = 0

        # Circuit related
        cir_property = self._circuit.name
        circuit_info = re.findall(r"\d+", cir_property)

        self._level = int(cir_property[-1])
        self._type = cir_property.split("+")[:-1][0]
        self._field = cir_property.split("+")[:-1][1]
        self._width = int(circuit_info[0])
        self._size = int(circuit_info[1])
        self._depth = int(circuit_info[2])
