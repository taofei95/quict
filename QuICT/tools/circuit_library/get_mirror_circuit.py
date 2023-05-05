from QuICT.core.circuit.circuit import Circuit
from QuICT.core.gate import *
import random


class MirrorCircuitBuilder:

    def _random_paulis(self, width:int):
        pauils_cgate = CompositeGate()
        qubits_indexes = list(range(width))
        for _ in range(width):
            gate_type = np.random.choice(PAULI_GATE_SET)
            gate = gate_builder(gate_type)
            gate_size = gate.controls + gate.targets
            gate & qubits_indexes[:gate_size] | pauils_cgate
            qubits_indexes = qubits_indexes[gate_size:]

        return pauils_cgate

    def _random_cliffords(self, width:int, pro:float, single:bool):
        single_typelist = CLIFFORD_GATE_SET[:-1]

        cliffords_cgate = CompositeGate()
        qubits_indexes = list(range(width))
        if single is False:
            prob = int((pro * width) / 2)
            assert 0 <= pro <= 1, "two qubits gate prob must be between 0 and 1"
            assert prob >= 1 , "Circuit must have one double qubits gate."
            count = 0
            for _ in range(width):
                cx_indxes = random.sample(qubits_indexes, 2)
                if abs(cx_indxes[0] - cx_indxes[1]) == 1:
                    CX & cx_indxes | cliffords_cgate
                    count += 1
                    qubits_indexes = list(set(qubits_indexes) - set(cx_indxes))
                if count == prob:
                    break
            if len(qubits_indexes) >= 1:
                for single_idx in qubits_indexes:
                    gate_type = np.random.choice(single_typelist)
                    gate = gate_builder(gate_type)
                    gate & single_idx | cliffords_cgate
        else:
            for _ in range(width):
                gate_type = np.random.choice(single_typelist)
                gate = gate = gate_builder(gate_type)
                gate_size = gate.controls + gate.targets
                gate & qubits_indexes[:gate_size] | cliffords_cgate
                qubits_indexes = qubits_indexes[gate_size:]

        return cliffords_cgate

    def build_mirror_circuit(self, width:int, rand_unit:int, pro:float):
        """Get mirror circuit for benchmark.

        Args:
            width (int): The width of the circuit.
            rand_unit (int): The number of sampling units.
            pro (float): The probability of two qubits gate in circuit.

        Returns:
            (List[Circuit]): Return the mirror circuit.

        """
        cir = Circuit(width)

        # A layer of H gate
        H | cir

        # single qubit cliffords
        single_clifford_gate = self._random_cliffords(width, pro, single=True)
        single_clifford_gate | cir

        # unit circuits obtain random paulis and random cliffords
        inverse_group_gate = []
        for _ in range(rand_unit):
            cgate_paulis = self._random_paulis(width)
            cgate_paulis | cir
            cgate_cliffords = self._random_cliffords(width, pro, single=False)
            cgate_cliffords | cir
            inverse_group_gate.append(cgate_cliffords.inverse())

        # random paulis
        c_paulis_gate = self._random_paulis(width)
        c_paulis_gate | cir

        # inverse unit circuits
        for c_group in reversed(inverse_group_gate):
            c_group | cir
            cgate = self._random_paulis(width)
            cgate | cir

        # inverse single qubit cliffords
        inverse_clifford_gate = single_clifford_gate.inverse()
        inverse_clifford_gate | cir

        # A layer of measure gate
        Measure | cir

        return cir

    def mirror_circuit_transform(circuit):
        inverse_list = []
        cgate = circuit.to_compositegate()
        for c in cgate:
            inverse_list.append(c)
        for inv in reversed(inverse_list):
            inv | circuit

        return circuit

if __name__ == "__main__":
    cir = MirrorCircuitBuilder().build_mirror_circuit(width=5, rand_unit=2, pro=0.8)
    cir.draw(filename="mirror_circuit")