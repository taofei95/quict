import cupy as cp
import random
import numpy as np
import torch
import time

from QuICT.algorithm.quantum_machine_learning.utils.gate_tensor import *
from QuICT.core.gate import *
from QuICT.ops.utils import LinAlgLoader


# class ApplyGate(torch.autograd.Function):
#     @staticmethod
#     # 第一个是ctx，第二个是input，其他是可选参数。
#     def forward(ctx, input, weight, bias=None):
#         ctx.save_for_backward(input, weight, bias)
#         output = input.mm(weight.t())  # torch.t()方法，对2D tensor进行转置
#         if bias is not None:
#             # expand_as(tensor)等价于expand(tensor.size()), 将原tensor按照新的size进行扩展
#             output += bias.unsqueeze(0).expand_as(output)
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         # grad_output为反向传播上一级计算得到的梯度值
#         # input, weight, bias = ctx.saved_variables
#         input, weight, bias = ctx.saved_tensors
#         grad_input = grad_weight = grad_bias = None
#         # 分别代表输入,权值,偏置三者的梯度
#         # 判断三者对应的Variable是否需要进行反向求导计算梯度
#         if ctx.needs_input_grad[0]:
#             grad_input = grad_output.mm(weight)
#         if ctx.needs_input_grad[1]:
#             grad_weight = grad_output.t().mm(input)
#         if bias is not None and ctx.needs_input_grad[2]:
#             grad_bias = grad_output.sum(0).squeeze(0)

#         return grad_input, grad_weight, grad_bias


class Ansatz:
    """The Ansatz class, which is similar to Circuit but support the auto-grad of Pytorch."""

    @property
    def n_qubits(self):
        return self._n_qubits

    @property
    def device(self):
        return self._device

    @property
    def gates(self):
        return self._gates

    def __init__(self, n_qubits, circuit=None, device=torch.device("cuda:0")):
        """Initialize an empty Ansatz or from a Circuit.

        Args:
            n_qubits (int): The number of qubits.
            circuit (Circuit, optional): Initialize an Ansatz from a Circuit. Defaults to None.
            device (torch.device, optional): The device to which the Ansatz is assigned.
                Defaults to torch.device("cuda:0").
        """
        self._n_qubits = n_qubits if circuit is None else circuit.width()
        self._device = device
        self._gates = [] if circuit is None else self._gate_to_tensor(circuit.gates)
        self._algorithm = (
            None
            if device.type == "cpu"
            else LinAlgLoader(
                device="GPU",
                enable_gate_kernel=True,
                enable_multigpu_gate_kernel=False,
            )
        )

    def __add__(self, other):
        """Add the gates of an ansatz into the ansatz."""
        ansatz = Ansatz(
            n_qubits=max(self._n_qubits, other._n_qubits), device=self._device
        )
        for gate in self._gates:
            gate.to(self._device)
            gate.update_name("ansatz", len(ansatz._gates))
            ansatz._gates.append(gate)
        for other_gate in other._gates:
            other_gate.to(self._device)
            other_gate.update_name("ansatz", len(ansatz._gates))
            ansatz._gates.append(other_gate)
        return ansatz

    def _gate_to_tensor(self, gates):
        """Copy the Circuit gates to Ansatz tensor gates."""
        gates_tensor = []
        for gate in gates:
            gate_tensor = BasicGateTensor(
                gate.controls, gate.targets, gate.params, gate.type, device=self._device
            )
            gate_tensor.pargs = torch.tensor(copy.deepcopy(gate.pargs)).to(self._device)
            gate_tensor.targs = copy.deepcopy(gate.targs)
            gate_tensor.cargs = copy.deepcopy(gate.cargs)
            gate_tensor.matrix = torch.from_numpy(gate.matrix).to(self._device)
            gate_tensor.assigned_qubits = copy.deepcopy(gate.assigned_qubits)
            gate_tensor.update_name(gate.assigned_qubits[0].id)
            gates_tensor.append(gate_tensor)

        return gates_tensor

    def add_gate(self, gate, act_bits: Union[int, list] = None):
        """Add a gate into the ansatz.

        Args:
            gate (BasicGateTensor): The tensor quantum gate.
            act_bits (Union[int, list], optional): The targets the gate acts on.
                Defaults to None, which means the gate will act on each qubit of the ansatz.
        """

        assert isinstance(gate.type, GateType)
        if gate.type == GateType.unitary:
            assert isinstance(gate.matrix, torch.Tensor)
            assert self._gate_validation(gate)

        if act_bits is None:
            for qid in range(self._n_qubits):
                new_gate = gate.to(self._device)
                new_gate.targs = [qid]
                new_gate.update_name("ansatz", len(self._gates))
                self._gates.append(new_gate)
        else:
            new_gate = gate.to(self._device)
            if isinstance(act_bits, int):
                new_gate.targs = act_bits
            else:
                assert len(act_bits) == new_gate.controls + new_gate.targets
                new_gate.cargs = act_bits[: new_gate.controls]
                new_gate.targs = act_bits[new_gate.controls :]
            new_gate.update_name("ansatz", len(self._gates))
            self._gates.append(new_gate)

    def _gate_validation(self, gate):
        """Validate the gate."""

        gate_matrix = gate.matrix
        shape = gate_matrix.shape
        log2_shape = int(np.ceil(np.log2(shape[0])))

        return (
            shape[0] == shape[1]
            and shape[0] == (1 << log2_shape)
            and torch.allclose(
                torch.eye(shape[0], dtype=gate.precision).to(self._device),
                torch.mm(gate_matrix, gate_matrix.T.conj()).to(self._device),
            )
        )

    def _apply_normal_matrix(self, gate: BasicGateTensor, default_parameters):
        # Get gate's parameters
        assert gate.matrix_type == MatrixType.normal
        args_num = gate.controls + gate.targets
        gate_args = gate.cargs + gate.targs
        matrix = cp.from_dlpack(gate.matrix.detach())

        # Deal with 1-qubit normal gate e.g. H
        if args_num == 1:
            index = self._n_qubits - 1 - gate_args[0]
            self._algorithm.normal_targ(index, matrix, *default_parameters)
        elif args_num == 2:  # Deal with 2-qubits control normal gate e.g. CH
            if gate.controls == gate.targets:
                c_index = self._n_qubits - 1 - gate.carg
                t_index = self._n_qubits - 1 - gate.targ
                self._algorithm.normal_ctargs(
                    c_index, t_index, matrix, *default_parameters
                )
            elif gate.targets == 2:  # Deal with 2-qubits unitary gate
                indexes = [self._n_qubits - 1 - index for index in gate_args]
                self._algorithm.normal_targs(indexes, matrix, *default_parameters)
            else:
                raise KeyError("Quantum gate cannot only have control qubits.")

    def _apply_diagonal_matrix(self, gate: BasicGateTensor, default_parameters):
        # Get gate's parameters
        assert gate.matrix_type in [MatrixType.diagonal, MatrixType.diag_diag]
        args_num = gate.controls + gate.targets
        gate_args = gate.cargs + gate.targs
        matrix = cp.from_dlpack(gate.matrix.detach())

        # Deal with 1-qubit diagonal gate, e.g. Rz
        if args_num == 1:
            index = self._n_qubits - 1 - gate.targ
            self._algorithm.diagonal_targ(index, matrix, *default_parameters)
        elif args_num == 2:  # Deal with 2-qubit diagonal gate, e.g. CRz
            if gate.controls == gate.targets:
                c_index = self._n_qubits - 1 - gate.carg
                t_index = self._n_qubits - 1 - gate.targ
                self._algorithm.diagonal_ctargs(
                    c_index, t_index, matrix, *default_parameters
                )
            elif gate.targets == 2:
                indexes = [self._qubits - 1 - index for index in gate_args]
                self._algorithm.diagonal_targs(indexes, matrix, *default_parameters)
            else:
                raise KeyError("Quantum gate cannot only have control qubits.")
        else:  # [CCRz]
            c_indexes = [self._n_qubits - 1 - carg for carg in gate.cargs]
            t_index = self._n_qubits - 1 - gate.targ
            self._algorithm.diagonal_more(
                c_indexes, t_index, matrix, *default_parameters
            )
    
    def _apply_swap_matrix(self, gate: BasicGateTensor, default_parameters):
        # Get gate's parameters
        assert gate.matrix_type == MatrixType.swap
        args_num = gate.controls + gate.targets
        gate_args = gate.cargs + gate.targs
        default_parameters = (self._vector, self._qubits, self._sync)

        if args_num == 1:       # Deal with X Gate
            index = self._qubits - 1 - gate.targ
            self._algorithm.swap_targ(
                index,
                *default_parameters
            )
        elif args_num == 2:     # Deal with Swap Gate
            t_indexes = [self._qubits - 1 - targ for targ in gate_args]
            self._algorithm.swap_targs(
                t_indexes,
                *default_parameters
            )
        else:   # CSwap
            t_indexes = [self._qubits - 1 - targ for targ in gate.targs]
            c_index = self._qubits - 1 - gate.carg
            self._algorithm.swap_tmore(
                t_indexes,
                c_index,
                *default_parameters
            )

    def _apply_gate_gpu(self, state: torch.Tensor, gate: BasicGateTensor):
        assert state.is_cuda, "Must use GPU."
        matrix_type = gate.matrix_type
        gate_type = gate.type
        cupy_state = cp.from_dlpack(state.detach())

        # Deal with quantum gate with more than 3 qubits.
        if (
            gate_type in [GateType.unitary, GateType.qft, GateType.iqft]
            and gate.targets >= 3
        ):
            matrix = cp.from_dlpack(gate.matrix.to(self._device).detach())
            matrix = matrix.reshape(
                1 << (gate.controls + gate.targets), 1 << (gate.controls + gate.targets)
            )
            cupy_state = self._algorithm.matrix_dot_vector(
                cupy_state, self._n_qubits, matrix, gate.cargs + gate.targs, True
            )
            state = torch.from_dlpack(cupy_state)
            return state

        default_parameters = (cupy_state, self._n_qubits, True)
        # [H, SX, SY, SW, U2, U3, Rx, Ry] 2-bits [CH, ] 2-bits[targets] [unitary]
        if matrix_type == MatrixType.normal:
            self._apply_normal_matrix(gate, default_parameters)
        # [Rz, Phase], 2-bits [CRz], 3-bits [CCRz]
        elif matrix_type in [MatrixType.diagonal, MatrixType.diag_diag]:
            self._apply_diagonal_matrix(gate, default_parameters)
        # [X] 2-bits [swap] 3-bits [CSWAP]
        elif matrix_type == MatrixType.swap:
            self.apply_swap_matrix(gate)
        # [Y] 2-bits [CX, CY] 3-bits: [CCX]
        elif matrix_type == MatrixType.reverse:
            self.apply_reverse_matrix(gate)
        # [S, sdg, Z, U1, T, tdg] # 2-bits [CZ, CU1]
        elif matrix_type == MatrixType.control:
            self.apply_control_matrix(gate)
        # [FSim]
        elif matrix_type == MatrixType.ctrl_normal:
            t_indexes = [self._qubits - 1 - targ for targ in gate.targs]
            matrix = self._get_gate_matrix(gate)
            self._algorithm.ctrl_normal_targs(t_indexes, matrix, *default_parameters)
        # [Rxx, Ryy]
        elif matrix_type == MatrixType.normal_normal:
            t_indexes = [self._qubits - 1 - targ for targ in gate.targs]
            matrix = self._get_gate_matrix(gate)
            self._algorithm.normal_normal_targs(t_indexes, matrix, *default_parameters)
        # [Measure, Reset]
        elif gate_type in [GateType.measure, GateType.reset]:
            index = self._qubits - 1 - gate.targ
            prob = self.get_measured_prob(index).get()
            self.apply_specialgate(index, gate_type, prob)
        # [Perm]
        elif gate_type == GateType.perm:
            args = gate.cargs + gate.targs
            if len(args) == self._qubits:
                mapping = np.array(gate.pargs, dtype=np.int32)
            else:
                mapping = np.arange(self._qubits, dtype=np.int32)
                for idx, parg in enumerate(gate.pargs):
                    mapping[args[idx]] = args[parg]

            self._algorithm.VectorPermutation(
                self._vector, mapping, changeInput=True, gpu_out=False, sync=self._sync
            )
        # unsupported quantum gates
        else:
            raise KeyError(f"Unsupported Gate: {gate_type}")

        state = torch.from_dlpack(cupy_state)
        return state

    def _apply_gate_cpu(self, state, gate_tensor, act_bits):
        """Apply a tensor gate to a state vector.

        Args:
            state (torch.Tensor): The initial state vector.
            gate_tensor (BasicGateTensor): The tensor quantum gate.
            act_bits (list): The targets the gate acts on.

        Returns:
            torch.Tensor: The state vector.
        """
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
                len(_act_idx), dtype=torch.int64
            ).to(self._device)

            # Step 4: Apply the gate on the action indices of the state.
            action_state = state.index_select(0, act_idx).reshape((act_idx.shape[0], 1))
            action_result = torch.mm(gate_tensor, action_state).reshape(
                act_idx.shape[0]
            )

            # Step 5: Refill the state vector according to the action indices.
            for i in range(len(act_idx)):
                state[act_idx[i]] = action_result[i]
        cp.cuda.Device().synchronize()

        return state

    def _apply_measuregate(self, qid, state):
        bits_idx = [1 << i for i in range(self._n_qubits)]
        qid_idx = 1 << qid
        offset = list(set(bits_idx) - (set([qid_idx])))
        idx_0 = [0]
        idx_1 = [qid_idx]
        for i in range(len(offset)):
            for j in range(len(idx_1)):
                idx_0.append(offset[i] + idx_0[j])
                idx_1.append(offset[i] + idx_1[j])

        # Calculate probabilities
        prob_0 = torch.sum(torch.abs(state[idx_0]) * torch.abs(state[idx_0]))

        _0 = random.random() < prob_0
        if _0:
            # The measured state of the qubit is |0>.
            alpha = 1 / torch.sqrt(prob_0)
            for idx in idx_1:
                state[idx] = 0
            for idx in idx_0:
                state[idx] *= alpha

        else:
            # The measured state of the qubit is |1>.
            alpha = 1 / torch.sqrt(1 - prob_0)
            for idx in idx_0:
                state[idx] = 0
            for idx in idx_1:
                state[idx] *= alpha

        return state, [prob_0, 1 - prob_0]

    def forward(self, state_vector=None):
        """The Forward Propagation process of an ansatz.

        Args:
            state_vector (np.array/torch.Tensor, optional): The initial state vector.
                Defaults to None, which means |0>.

        Returns:
            torch.Tensor: The state vector.
        """
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
            # Measure gate
            if gate.type == GateType.measure:
                qid = self._n_qubits - 1 - gate.targ
                state, prob = self._apply_measuregate(qid, state)
                return state, prob
            # CPU
            if self._device.type == "cpu":
                gate_tensor = gate.matrix.to(self._device)
                act_bits = gate.cargs + gate.targs
                state = self._apply_gate_cpu(state, gate_tensor, act_bits)
            # GPU
            else:
                # Non-parametric gates or gates with untrainable pargs
                if gate.params == 0 or not gate.pargs.requires_grad:
                    state = self._apply_gate_gpu(state, gate)
                else:
                    raise ValueError

        return state, None


if __name__ == "__main__":
    from QuICT.simulation.state_vector import ConstantStateVectorSimulator
    from QuICT.algorithm.quantum_machine_learning.utils.gate_tensor import *
    from QuICT.core import Circuit
    from QuICT.core.gate import *
    import time

    ansatz = Ansatz(16)
    ansatz.add_gate(H_tensor)  # 0.0007s
    s = time.time()
    state = ansatz.forward()  # 0.02s
    print(time.time() - s)

    circuit = Circuit(16)
    H | circuit  # 0.0006s
    simulator = ConstantStateVectorSimulator()
    s = time.time()
    sv = simulator.run(circuit)
    print(time.time() - s)  # 0.003s

    ansatz = Ansatz(16)
    ansatz.add_gate(H_tensor)  # 0.0007s
    s = time.time()
    state = ansatz.forward()  # 0.02s
    print(time.time() - s)

