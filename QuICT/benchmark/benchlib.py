import numpy as np
import re

from QuICT.core.circuit.circuit import Circuit


class BenchLib:
    """ A data structure for storing benchmark information. """
    @property
    def circuit(self) -> Circuit:
        """ Return the circuits of QuantumMachinebenchmark. """
        return self._circuit

    @property
    def machine_amp(self) -> np.array:
        """ Return the quantum machine sample result of circuits. """
        return self._machine_amp

    @machine_amp.setter
    def machine_amp(self, machine_amp:np.array):
        self._machine_amp = machine_amp

    @property
    def benchmark_score(self) -> float:
        """ Return the general benchmark score of each circuit. """
        return self._benchmark_score

    @benchmark_score.setter
    def benchmark_score(self, benchmark_score:float):
        self._benchmark_score = benchmark_score

    @property
    def type(self) -> str:
        """ Return the field of circuits. """
        self._type = self._circuit.name.split("+")[:-1][0]
        return self._type

    @property
    def field(self) -> str:
        """ Return the field of circuits. """
        self._field = self._circuit.name.split("+")[:-1][1]
        return self._field

    @property
    def level(self) -> list:
        """ Return the level of circuits. """
        self._level = int(self._circuit.name[-1])
        return self._level

    @property
    def width(self):
        """ Return the qubit number of circuit. """
        self._width = re.findall(r"\d+", self.circuit.name)[0]
        return self._width

    @property
    def size(self):
        """ Return the gate number of circuit. """
        self._size = re.findall(r"\d+", self.circuit.name)[1]
        return self._size

    @property
    def depth(self):
        """ Return the depth of circuit. """
        self._depth = re.findall(r"\d+", self.circuit.name)[2]
        return self._depth

    @property
    def value(self):
        """ Return the value of benchmark circuit. """
        cir_value = float(re.findall(r"\d+(?:\.\d+)?", self.circuit.name)[3])
        return cir_value

    @property
    def qv(self) -> list:
        """ Return the quantum volume of circuit. """
        cir_attribute = re.findall(r"\d+", self.circuit.name)
        self._qv = (2 ** min(int(cir_attribute[0]), int(cir_attribute[2])))
        return self._qv

    @property
    def fidelity(self) -> str:
        """ Return the fidelity of circuit. """
        if self._type != "algorithm" or self._field == "adder":
            self._fidelity = self._machine_amp[0]
        else:
            width = self.width
            if self._field == "qft":
                p, q = self._machine_amp, [float(1 / (2 ** int(width)))] * (2 ** int(width))
                self._fidelity = self._alg_cir_entropy(p, q)
            elif self._field == "cnf":
                self._fidelity = self._machine_amp[8]
            elif self._field == "qnn":
                point1 = self._machine_amp[0] + self._machine_amp[(2 ** width) / 2]
                point2 = self._machine_amp[3] + self._machine_amp[(2 ** width) / 2 + 3]
                self._fidelity = max(point1, point2)
            elif self._field == "quantum_walk":
                self._fidelity = self._machine_amp[3] + self._machine_amp[-2]
            elif self._field == "vqe":
                index = '1' * int(width / 2)
                if width % 2 == 1:
                    index += '0' * (int(width / 2) + 1)
                else:
                    index += '0' * int(width / 2)
                self._fidelity = self._machine_amp[int(index, 2)]

        return round(self._fidelity, 4)

    def _alg_cir_entropy(self, p, q):
        def normalization(data):
            data = np.array(data)
            data = data/np.sum(data)

            return data

        p = abs(normalization(p))
        q = abs(normalization(q))

        import math

        sum=0.0
        delta=1e-7
        for x in map(lambda y, p:(1 - y) * math.log(1 - p + delta) + y * math.log(p + delta), p, q):
            sum+=x
        cross_entropy = -sum / len(p)

        return cross_entropy

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
        self._fidelity = 0

        # Circuit related
        self._level = 0
        self._type = 0
        self._field = 0
        self._width = 0
        self._size = 0
        self._depth = 0
