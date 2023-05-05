from scipy.stats import unitary_group
import random
from QuICT.core.gate import *
from QuICT.core import Circuit
from QuICT.qcda.synthesis.unitary_decomposition import CartanKAKDecomposition


class QuantumVolumnCircuitBuilder:

    def _SU4_group(self, width:int):
        # build SU(4) group gates
        cgate = CompositeGate()

        U = unitary_group.rvs(2 ** 2)
        KAK_decomposition = CartanKAKDecomposition()
        compositeGate = KAK_decomposition.execute(U)
        compositeGate | cgate

        return cgate

    def _U3_group(self, width):
        # If width is odd, there is a qubit in the free and others have U3 gate. 
        # On the contrary, all qubits has U3 gate.
        cgate = CompositeGate()
        width_indexes = list(range(width))
        if width % 2 == 0:
            for i in width_indexes:
                U3 | cgate(i)
        else:
            extra_qubit = random.choice(width_indexes)
            width_indexes.remove(extra_qubit)
            for i in width_indexes:
                U3 | cgate(i)
        return cgate

    def build_qv_circuit(self, width:int, depth:int):
        circuit = Circuit(width)
        width_indexes = list(range(width))

        for _ in range(depth):
            i = 0
            for _ in range(int(width / 2)):
                # A layer of U3
                random_U3_circuit = self._U3_group(width)
                random_U3_circuit | circuit
                # A group of SU(4)
                group_circuit = self._SU4_group(width)
                group_circuit | circuit([width_indexes[i], width_indexes[i + 1]])
                i += 2

        Measure | circuit

        return circuit

if __name__ == "__main__":
    circuit = QuantumVolumnCircuitBuilder().build_qv_circuit(5, 1)
    circuit.draw(filename="qv_circuit")