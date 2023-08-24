from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.core.utils.gate_type import GateType


class Simulationbenchmark:
    """ A benchmarking framework for quantum simulation."""

    def _random_circuit(self, bench_scale):
        cirs_group = []
        gate_prob_list = {
            "small": [[3, 5, 10], [10, 50, 100]],
            "medium": [[10, 15, 20], [500, 1000, 5000]],
            "large": [[20, 25, 30], [10000, 50000, 100000]]
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
                cir.random_append(qubit * gate_prob, typelist=one_qubit+two_qubits, probabilities=prob_list, seed=10)
                cir.name = "+".join(["simbench", bench_scale, f"w{cir.width()}_s{cir.size()}_d{cir.depth()}"])
                cirs_group.append(cir)
        return cirs_group

    def get_data(self):
        """Get the data to be benchmarked."""
        circuits_list = []

        _bench_scale = ["small", "medium", "large"]
        # range time
        import time
        d_time = time.time()

        for index in range(len(_bench_scale)):
            scale = _bench_scale[index]
            # circuits with different probabilities of cnot
            circuits_list.extend(self._random_circuit(scale))
            n_time = time.time()
            # determine if the current time is within the range time
            if n_time - d_time > 10:
                return circuits_list
            else:
                continue