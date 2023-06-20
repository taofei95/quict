from scipy.stats import unitary_group
import random
from QuICT.core.gate import *
from QuICT.core import Circuit
from QuICT.qcda.synthesis.unitary_decomposition import CartanKAKDecomposition


class QuantumVolumnCircuitBuilder:

    def _SU4_group(self):
        # build SU(4) group gates
        cgate = CompositeGate()

        U = unitary_group.rvs(2 ** 2)
        KAK_decomposition = CartanKAKDecomposition()
        compositeGate = KAK_decomposition.execute(U)
        compositeGate | cgate

        return cgate

    def build_qv_circuit(self, width:int, size:int):
        circuit = Circuit(width)
        for _ in range(width):
            while circuit.size() < size:
                # A group of SU(4)
                width_indexes = list(range(width))
                # If width is odd, there is a qubit in the free and others have SU(4) group. 
                # On the contrary, all qubits has SU(4) group.
                if width % 2 != 0:
                    extra_qubit = random.choice(width_indexes)
                    width_indexes.remove(extra_qubit)
                group_circuit = self._SU4_group()
                group_circuit | circuit(random.sample(width_indexes, 2))

        Measure | circuit

        return circuit

if __name__ == "__main__":
    circuit = QuantumVolumnCircuitBuilder().build_qv_circuit(3, 40)
    print(circuit.size())
    circuit.draw(filename="qv_circuit")