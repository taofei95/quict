import ssl
from tkinter import N
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
        state.scatter_(0, act_idx, action_result)

        return state

    def forward(self, state_vector=None):
        if state_vector is None:
            state = torch.zeros(1 << self._n_qubits, dtype=torch.complex128).to(
                self._device
            )
            state[0] = 1

        gates = self._circuit.gates
        for gate in gates:
            gate_tensor = torch.from_numpy(gate.matrix).to(self._device)
            act_bits = gate.cargs + gate.targs
            state = self._apply_gate(state, gate_tensor, act_bits)

        return state


if __name__ == "__main__":
    from QuICT.simulation.state_vector import ConstantStateVectorSimulator

    circuit = Circuit(5)
    HH = np.kron(H.matrix, H.matrix)
    HHH = np.kron(HH, H.matrix)
    HHHH = np.kron(HHH, H.matrix)
    HHHH = Unitary(HHHH)
    HHHH | circuit([3, 1, 0, 2])
    ansatz = Ansatz(circuit)
    state = ansatz.forward()
    print(np.array(state.cpu()))

    simulator = ConstantStateVectorSimulator()
    sv = simulator.run(circuit)
    print(sv)

