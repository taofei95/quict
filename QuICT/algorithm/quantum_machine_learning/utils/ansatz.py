import torch
import numpy as np
from QuICT.core import Circuit
from QuICT.core.gate import *


class Ansatz:
    def __init__(self, circuit: Circuit, device=torch.device("cuda:0")):
        self._circuit = circuit
        self._n_qubits = self._circuit.width()
        self._device = device

    def _apply_gate(self, state, gate_tensor, act_bits):
        # Step 1: Relabel the qubits and calculate their corresponding index values.
        act_bits_idx = list(np.zeros(len(act_bits), dtype=np.int32))
        for i in range(len(act_bits)):
            act_bits_idx[i] = 1 << (self._n_qubits - 1 - act_bits[i])
        act_bits_idx.reverse()

        # Step 2: Get action indices.
        act_idx = [0]
        for i in range(len(act_bits_idx)):
            for j in range(len(act_idx)):
                act_idx.append(act_bits_idx[i] + act_idx[j])
        act_idx = torch.tensor(act_idx).to(self._device)
        action_state = state.index_select(0, act_idx).reshape((act_idx.shape[0], 1))

        # Step 3: Apply the gate on the action indices of the state.
        action_result = torch.mm(gate_tensor, action_state).reshape(act_idx.shape[0])

        # Step 4: Refill the state vector according to the action indices.
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
                state = state_vector
        assert state.shape[0] == 1 << self._n_qubits

        gates = self._circuit.gates
        for gate in gates:
            gate_tensor = torch.from_numpy(gate.matrix).to(self._device)
            act_bits = gate.cargs + gate.targs
            state = self._apply_gate(state, gate_tensor, act_bits)

        return state


if __name__ == "__main__":
    from QuICT.simulation.state_vector import ConstantStateVectorSimulator

    state = np.array(
        [np.sqrt(3) / 3, 1 / 2, 1 / 3, np.sqrt(11) / 6], dtype=np.complex128
    )
    print("init state ", state.real)
    circuit = Circuit(2)
    H | circuit(1)
    ansatz = Ansatz(circuit)
    sv = ansatz.forward(state)
    print(np.array(sv.cpu()).real)

    simulator = ConstantStateVectorSimulator()
    sv = simulator.run(circuit, state)
    print(sv.real)

