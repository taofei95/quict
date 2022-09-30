from ssl import HAS_ECDH
import torch
import numpy as np
from QuICT.core import Circuit
from QuICT.core.gate import *


class Ansatz:
    def __init__(self, n_qubits, circuit=None, device=torch.device("cuda:0")):
        self._circuit = circuit
        self._n_qubits = n_qubits if circuit is None else circuit.width()
        self._gates = [] if circuit is None else circuit.gates
        self._device = device

    def add_gate(self, gate, act_bits: Union[int, list] = None):
        assert isinstance(gate.type, GateType)
        if act_bits is None:
            for qid in range(self._n_qubits):
                new_gate = gate.copy()
                new_gate.targs = [qid]
                new_gate.update_name("ansatz", len(self._gates))
                self._gates.append(new_gate)
        else:
            if isinstance(act_bits, int):
                gate.targs = act_bits
            else:
                assert len(act_bits) == gate.controls + gate.targets
                gate.cargs = act_bits[: gate.controls]
                gate.targs = act_bits[gate.controls :]
            gate.update_name("ansatz", len(self._gates))
            self._gates.append(gate)

    def _apply_gate(self, state, gate_tensor, act_bits):
        # Step 1: Relabel the qubits and calculate their corresponding index values.
        act_bits = [self._n_qubits - 1 - act_bits[i] for i in range(len(act_bits))]
        bits_idx = [1 << i for i in range(self._n_qubits)]
        act_bits_idx = list(np.array(bits_idx)[act_bits])
        act_bits_idx.reverse()
        inact_bits_idx = list(set(bits_idx) - (set(act_bits_idx)))

        # Step 2: Get the first set of action indices.
        _act_idx = [0]
        for i in range(len(act_bits_idx)):
            for j in range(len(_act_idx)):
                _act_idx.append(act_bits_idx[i] + _act_idx[j])
        _act_idx = torch.tensor(_act_idx).to(self._device)

        # Step 3: Get the following action indices.
        offsets = [0]
        for i in range(len(inact_bits_idx)):
            for j in range(len(offsets)):
                offsets.append(inact_bits_idx[i] + offsets[j])

        for offset in offsets:
            act_idx = _act_idx + offset * torch.ones(
                len(_act_idx), dtype=torch.int32
            ).to(self._device)
            # Step 4: Apply the gate on the action indices of the state.
            action_state = state.index_select(0, act_idx).reshape((act_idx.shape[0], 1))
            action_result = torch.mm(gate_tensor, action_state).reshape(
                act_idx.shape[0]
            )

            # Step 5: Refill the state vector according to the action indices.
            for i in range(len(act_idx)):
                state[act_idx[i]] = action_result[i]

        return state

    def forward(self, state_vector=None):
        if state_vector is None:
            state = torch.zeros(1 << self._n_qubits, dtype=torch.complex128).to(
                self._device
            )
            state[0] = 1
        else:
            if isinstance(state_vector, np.ndarray):
                state = torch.from_numpy(state_vector).to(self._device)
            else:
                state = state_vector.clone()
        assert state.shape[0] == 1 << self._n_qubits

        gates = self._gates if self._circuit is None else self._circuit.gates
        for gate in gates:
            print(gate)
            gate_tensor = torch.from_numpy(gate.matrix).to(self._device)
            act_bits = gate.cargs + gate.targs
            state = self._apply_gate(state, gate_tensor, act_bits)

        return state


if __name__ == "__main__":
    from QuICT.simulation.state_vector import ConstantStateVectorSimulator

    def random_pauli_str(n_items, n_qubits):
        pauli_str = []
        coeffs = np.random.rand(n_items)
        for i in range(n_items):
            pauli = [coeffs[i]]
            for qid in range(n_qubits):
                flag = np.random.randint(0, 5)
                if flag == 0:
                    pauli.append("X" + str(qid))
                elif flag == 1:
                    pauli.append("Y" + str(qid))
                elif flag == 2:
                    pauli.append("Z" + str(qid))
                elif flag == 3:
                    pauli.append("I" + str(qid))
                elif flag == 4:
                    continue
            pauli_str.append(pauli)
        return pauli_str

    def random_state(n_qubits):
        state = np.random.randn(1 << n_qubits)
        state /= sum(abs(state))
        state = abs(state) ** 0.5
        state = state.astype(np.complex128)
        return state

    state = random_state(5)
    circuit = Circuit(5)
    HH = np.kron(H.matrix, H.matrix)
    HH = Unitary(HH)
    HH | circuit([0, 1])
    # Rz(0.5) | circuit(2)
    # ansatz = Ansatz(5, circuit)
    # sv = ansatz.forward(state)

    ansatz2 = Ansatz(5)
    ansatz2.add_gate(H)
    sv = ansatz2.forward(state)
    # print(np.array(sv.cpu()).real)

    # simulator = ConstantStateVectorSimulator()
    # sv = simulator.run(circuit, state)
    # print(sv.real)
