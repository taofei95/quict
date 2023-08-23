from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.core.utils.gate_type import GateType


class Simulationbenchmark:
    """ A benchmarking framework for quantum circuits automated design."""

    def _random_circuit(self, bench_scale):
        cirs_group = []
        qubit_list = [5, 10, 15, 20, 25, 30]
        gate_prob_list = {
            "small": [5, 10, 50, 100,],
            "medium": [500, 1000, 5000],
            "large": [10000, 50000, 100000]
        }
        one_qubit = [GateType.h, GateType.rx, GateType.ry, GateType.rz, GateType.x, GateType.y, GateType.z]
        two_qubits = [GateType.cx, GateType.cz]

        for gate_prob in gate_prob_list[bench_scale]:
            for qubit in qubit_list:
                prob = 0.8
                len_s, len_d = len(one_qubit), len(two_qubits)
                prob = [prob / len_s] * len_s + [(1 - prob) / len_d] * len_d
                cir = Circuit(qubit)
                cir.random_append(qubit * gate_prob, typelist=one_qubit+two_qubits, probabilities=prob, seed=10)
                cir.name = "+".join(["simbench", bench_scale, f"w{cir.width()}_s{cir.size()}_d{cir.depth()}"])
                cirs_group.append(cir)
        return cirs_group

    def get_data(self):
        """Get the data to be benchmarked.
        """
        circuits_list = []

        _bench_scale = ["small", "medium", "large"]
        for scale in _bench_scale:
            # circuits with different probabilities of cnot
            circuits_list.extend(self._random_circuit(scale))

        return circuits_list
