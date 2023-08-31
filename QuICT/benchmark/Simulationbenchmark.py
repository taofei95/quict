from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.core.utils.gate_type import GateType


class Simulationbenchmark:
    """ A benchmarking framework for quantum simulation."""

    def _random_circuit(self, bench_scale):
        cirs_group = []
        gate_prob_list = {
            "medium": [[3, 5, 10], [10, 50, 100, 500]],
            "large": [[15, 2020, 25, 30], [1000, 5000, 10000, 50000]]
        }
        one_qubit = [GateType.h, GateType.rx, GateType.ry, GateType.rz, GateType.x, GateType.y, GateType.z]
        two_qubits = [GateType.cx, GateType.cz]
        prob = 0.8

        scale_index = gate_prob_list[bench_scale]
        for qubit in scale_index[0]:
            for gate_prob in scale_index[1]:
                len_s, len_d = len(one_qubit), len(two_qubits)
                prob_list = [prob / len_s] * len_s + [(1 - prob) / len_d] * len_d
                cir = Circuit(qubit)
                cir.random_append(qubit * gate_prob, typelist=one_qubit + two_qubits, probabilities=prob_list, seed=10)
                cir.name = "+".join(["simbench", bench_scale, f"w{cir.width()}_s{cir.size()}_d{cir.depth()}"])
                cirs_group.append(cir)
        return cirs_group

    def get_data(self):
        """Get the data to be benchmarked."""
        # range time
        import time
        d_time = time.time()

        # circuits with different probabilities of cnot
        medium_cirs = self._random_circuit("medium")
        large_cirs = self._random_circuit("large")
        # determine if the current time is within the range time

        return [medium_cirs, large_cirs]
