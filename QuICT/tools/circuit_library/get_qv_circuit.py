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





































# """Functions for creating circuits of the form used in quantum
# volume experiments as defined in https://arxiv.org/abs/1811.12926.

# Useful overview of quantum volume experiments:
# https://pennylane.ai/qml/demos/quantum_volume.html

# Cirq implementation of quantum volume circuits:
# cirq-core/cirq/contrib/quantum_volume/quantum_volume.py
# """

# from typing import Optional, Tuple, Sequence

# from numpy import random

# from cirq import decompose as cirq_decompose
# from cirq.circuits import Circuit
# from cirq.contrib.quantum_volume import (
#     generate_model_circuit,
#     compute_heavy_set,
# )
# from cirq.value import big_endian_int_to_bits


# from mitiq import QPROGRAM
# from mitiq import Bitstring
# from mitiq.interface import convert_from_mitiq

# def generate_model_circuit(
#     num_qubits: int, depth: int, *, random_state: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None
# ) -> cirq.Circuit:
#     """Generates a model circuit with the given number of qubits and depth.
#     The generated circuit consists of `depth` layers of random qubit
#     permutations followed by random two-qubit gates that are sampled from the
#     Haar measure on SU(4).
#     Args:
#         num_qubits: The number of qubits in the generated circuit.
#         depth: The number of layers in the circuit.
#         random_state: Random state or random state seed.
#     Returns:
#         The generated circuit.
#     """
#     # Setup the circuit and its qubits.
#     qubits = cirq.LineQubit.range(num_qubits)
#     circuit = cirq.Circuit()
#     random_state = cirq.value.parse_random_state(random_state)

#     # For each layer.
#     for _ in range(depth):
#         # Generate uniformly random permutation Pj of [0...n-1]
#         perm = random_state.permutation(num_qubits)

#         # For each consecutive pair in Pj, generate Haar random SU(4)
#         # Decompose each SU(4) into CNOT + SU(2) and add to Ci
#         for k in range(0, num_qubits - 1, 2):
#             permuted_indices = [int(perm[k]), int(perm[k + 1])]
#             special_unitary = cirq.testing.random_special_unitary(4, random_state=random_state)

#             # Convert the decomposed unitary to Cirq operations and add them to
#             # the circuit.
#             circuit.append(
#                 cirq.MatrixGate(special_unitary).on(
#                     qubits[permuted_indices[0]], qubits[permuted_indices[1]]
#                 )
#             )

#     # Don't measure all of the qubits at the end of the circuit because we will
#     # need to classically simulate it to compute its heavy set.
#     return circuit


# def generate_quantum_volume_circuit(
#     num_qubits: int,
#     depth: int,
#     decompose: bool = False,
#     seed: Optional[int] = None,
#     return_type: Optional[str] = None,
# ) -> Tuple[QPROGRAM, Sequence[Bitstring]]:
#     """Generate a quantum volume circuit with the given number of qubits and
#     depth.

#     The generated circuit consists of `depth` layers of random qubit
#     permutations followed by random two-qubit gates that are sampled from the
#     Haar measure on SU(4).

#     Args:
#         num_qubits: The number of qubits in the generated circuit.
#         depth: The number of qubits in the generated circuit.
#         decompose: Recursively decomposes the randomly sampled (numerical)
#             unitary matrix gates into simpler gates.
#         seed: Seed for generating random circuit.
#         return_type: String which specifies the type of the returned
#             circuits. See the keys of ``mitiq.SUPPORTED_PROGRAM_TYPES``
#             for options. If ``None``, the returned circuits have type
#             ``cirq.Circuit``.

#     Returns:
#         A quantum volume circuit acting on ``num_qubits`` qubits.
#         A list of the heavy bitstrings for the returned circuit.
#     """
#     random_state = random.RandomState(seed)
#     circuit = generate_model_circuit(
#         num_qubits, depth, random_state=random_state
#     )
#     heavy_bitstrings = compute_heavy_bitstrings(circuit, num_qubits)

#     if decompose:
#         # Decompose random unitary gates into simpler gates.
#         circuit = Circuit(cirq_decompose(circuit))

#     return_type = "cirq" if not return_type else return_type
#     return convert_from_mitiq(circuit, return_type), heavy_bitstrings



# def compute_heavy_bitstrings(
#     circuit: Circuit,
#     num_qubits: int,
# ) -> Sequence[Bitstring]:
#     """Classically compute the heavy bitstrings of the provided circuit.

#     The heavy bitstrings are defined as the output bit-strings that have a
#     greater than median probability of being generated.

#     Args:
#         circuit: The circuit to classically simulate.

#     Returns:
#         A list containing the heavy bitstrings.
#     """
#     heavy_vals = compute_heavy_set(circuit)
#     # Convert base-10 ints to Bitstrings.
#     heavy_bitstrings = [
#         big_endian_int_to_bits(val, bit_count=num_qubits) for val in heavy_vals
#     ]
#     return heavy_bitstrings


