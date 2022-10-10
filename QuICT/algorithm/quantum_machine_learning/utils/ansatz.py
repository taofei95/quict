import torch
import numpy as np
from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.algorithm.quantum_machine_learning.utils.gate_tensor import *
from QuICT.algorithm.quantum_machine_learning.utils.gate_tensor import BasicGateTensor


class Ansatz:
    def __init__(self, n_qubits, circuit=None, device=torch.device("cuda:0")):
        self._circuit = circuit
        self._n_qubits = n_qubits if circuit is None else circuit.width()
        self._gates = [] if circuit is None else self._gate_to_tensor(circuit.gates)
        self._device = device

    def __add__(self, other):
        ansatz = Ansatz(
            n_qubits=max(self._n_qubits, other._n_qubits), device=self._device
        )
        for gate in self._gates:
            gate.update_name("ansatz", len(ansatz._gates))
            ansatz._gates.append(gate)
        for other_gate in other._gates:
            other_gate.update_name("ansatz", len(ansatz._gates))
            ansatz._gates.append(other_gate)
        return ansatz

    def _gate_to_tensor(self, gates):
        gates_tensor = []
        for gate in gates:
            gate_tensor = BasicGateTensor(
                gate.controls, gate.targets, gate.params, gate.type
            )
            gate_tensor.pargs = copy.deepcopy(gate.pargs)
            gate_tensor.targs = copy.deepcopy(gate.targs)
            gate_tensor.cargs = copy.deepcopy(gate.cargs)
            gate_tensor.matrix = torch.from_numpy(gate.matrix)
            gate_tensor.assigned_qubits = copy.deepcopy(gate.assigned_qubits)
            gate_tensor.update_name(gate.assigned_qubits[0].id)
            gates_tensor.append(gate_tensor)

        return gates_tensor

    def add_gate(self, gate, act_bits: Union[int, list] = None):
        assert isinstance(gate.type, GateType)
        assert isinstance(gate.matrix, torch.Tensor)
        assert self._gate_validation(gate)
        if act_bits is None:
            for qid in range(self._n_qubits):
                new_gate = gate.copy()
                new_gate.targs = [qid]
                new_gate.update_name("ansatz", len(self._gates))
                self._gates.append(new_gate)
        else:
            new_gate = gate.copy()
            if isinstance(act_bits, int):
                new_gate.targs = act_bits
            else:
                assert len(act_bits) == new_gate.controls + new_gate.targets
                new_gate.cargs = act_bits[: new_gate.controls]
                new_gate.targs = act_bits[new_gate.controls :]
            new_gate.update_name("ansatz", len(self._gates))
            self._gates.append(new_gate)

    def _gate_validation(self, gate):
        gate_matrix = gate.matrix
        shape = gate_matrix.shape
        log2_shape = int(np.ceil(np.log2(shape[0])))

        return (
            shape[0] == shape[1]
            and shape[0] == (1 << log2_shape)
            and torch.allclose(
                torch.eye(shape[0], dtype=gate.precision).to(self._device),
                torch.mm(gate_matrix, gate_matrix.T.conj()),
            )
        )

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

        for gate in self._gates:
            gate_tensor = gate.matrix.to(self._device)
            act_bits = gate.cargs + gate.targs
            state = self._apply_gate(state, gate_tensor, act_bits)

        return state


if __name__ == "__main__":
    from QuICT.simulation.state_vector import ConstantStateVectorSimulator

    state = np.array(
        [
            -0.2118 + 1.3275e-01j,
            0.0691 + 2.4027e-01j,
            0.0691 + 2.4027e-01j,
            0.2500 + 1.2772e-19j,
            0.0691 + 2.4027e-01j,
            0.2500 + 1.2772e-19j,
            0.2500 + 1.6824e-17j,
            0.0691 - 2.4027e-01j,
            0.0691 + 2.4027e-01j,
            0.2500 - 1.6824e-17j,
            0.2500 - 1.2772e-19j,
            0.0691 - 2.4027e-01j,
            0.2500 - 1.2772e-19j,
            0.0691 - 2.4027e-01j,
            0.0691 - 2.4027e-01j,
            -0.2118 - 1.3275e-01j,
        ]
    )
    simulator = ConstantStateVectorSimulator()
    circuit = Circuit(4)
    H | circuit
    Rx(0.4) | circuit
    sv = simulator.run(circuit)
    print(sv.real)
    print(sum(sv.real * sv.real))

    ansatz = Ansatz(4)
    ansatz.add_gate(H_tensor)
    ansatz.add_gate(Rx_tensor(0.4))
    sv = ansatz.forward()
    print(sv.real.cpu().numpy())
    print(sum(sv.real.cpu().numpy() * sv.real.cpu().numpy()))
