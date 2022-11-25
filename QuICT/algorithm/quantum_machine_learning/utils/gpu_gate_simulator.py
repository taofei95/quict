import cupy as cp
import torch
import random

from QuICT.ops.utils import LinAlgLoader
from QuICT.algorithm.quantum_machine_learning.utils.gate_tensor import *


def apply_normal_matrix(
    gate: BasicGateTensor, default_parameters, algorithm, n_qubits, fp
):
    # Get gate's parameters
    assert gate.matrix_type == MatrixType.normal
    args_num = gate.controls + gate.targets
    gate_args = gate.cargs + gate.targs
    matrix = (
        cp.from_dlpack(gate.matrix.clone())
        if fp
        else cp.from_dlpack(gate.gradient.clone())
    )

    # Deal with 1-qubit normal gate e.g. H
    if args_num == 1:
        index = n_qubits - 1 - gate_args[0]
        algorithm.normal_targ(index, matrix, *default_parameters)
    elif args_num == 2:  # Deal with 2-qubits control normal gate e.g. CH
        if gate.controls == gate.targets:
            c_index = n_qubits - 1 - gate.carg
            t_index = n_qubits - 1 - gate.targ
            algorithm.normal_ctargs(c_index, t_index, matrix, *default_parameters)
        elif gate.targets == 2:  # Deal with 2-qubits unitary gate
            indexes = [n_qubits - 1 - index for index in gate_args]
            algorithm.normal_targs(indexes, matrix, *default_parameters)
        else:
            raise KeyError("Quantum gate cannot only have control qubits.")


def apply_normal_normal_matrix(
    gate: BasicGateTensor, default_parameters, algorithm, n_qubits, fp
):
    assert gate.matrix_type == MatrixType.normal_normal
    t_indexes = [n_qubits - 1 - targ for targ in gate.targs]
    matrix = (
        cp.from_dlpack(gate.matrix.clone())
        if fp
        else cp.from_dlpack(gate.gradient.clone())
    )
    algorithm.normal_normal_targs(t_indexes, matrix, *default_parameters)


def apply_diagonal_normal_matrix(
    gate: BasicGateTensor, default_parameters, algorithm, n_qubits, fp
):
    assert gate.matrix_type == MatrixType.normal_normal
    t_indexes = [n_qubits - 1 - targ for targ in gate.targs]
    matrix = (
        cp.from_dlpack(gate.matrix.clone())
        if fp
        else cp.from_dlpack(gate.gradient.clone())
    )
    algorithm.diagonal_normal_targs(t_indexes, matrix, *default_parameters)


def apply_diagonal_matrix(
    gate: BasicGateTensor, default_parameters, algorithm, n_qubits, fp
):
    # Get gate's parameters
    assert gate.matrix_type in [MatrixType.diagonal, MatrixType.diag_diag]
    args_num = gate.controls + gate.targets
    gate_args = gate.cargs + gate.targs
    matrix = (
        cp.from_dlpack(gate.matrix.clone())
        if fp
        else cp.from_dlpack(gate.gradient.clone())
    )

    # Deal with 1-qubit diagonal gate, e.g. Rz
    if args_num == 1:
        index = n_qubits - 1 - gate.targ
        algorithm.diagonal_targ(index, matrix, *default_parameters)
    elif args_num == 2:  # Deal with 2-qubit diagonal gate, e.g. CRz
        if gate.controls == gate.targets:
            c_index = n_qubits - 1 - gate.carg
            t_index = n_qubits - 1 - gate.targ
            algorithm.diagonal_ctargs(c_index, t_index, matrix, *default_parameters)
        elif gate.targets == 2:
            indexes = [n_qubits - 1 - index for index in gate_args]
            algorithm.diagonal_targs(indexes, matrix, *default_parameters)
        else:
            raise KeyError("Quantum gate cannot only have control qubits.")
    else:  # [CCRz]
        c_indexes = [n_qubits - 1 - carg for carg in gate.cargs]
        t_index = n_qubits - 1 - gate.targ
        algorithm.diagonal_more(c_indexes, t_index, matrix, *default_parameters)


def apply_swap_matrix(gate: BasicGateTensor, default_parameters, algorithm, n_qubits):
    # Get gate's parameters
    assert gate.matrix_type == MatrixType.swap
    args_num = gate.controls + gate.targets
    gate_args = gate.cargs + gate.targs

    if args_num == 1:  # Deal with X Gate
        index = n_qubits - 1 - gate.targ
        algorithm.swap_targ(index, *default_parameters)
    elif args_num == 2:  # Deal with Swap Gate
        t_indexes = [n_qubits - 1 - targ for targ in gate_args]
        algorithm.swap_targs(t_indexes, *default_parameters)
    else:  # CSwap
        t_indexes = [n_qubits - 1 - targ for targ in gate.targs]
        c_index = n_qubits - 1 - gate.carg
        algorithm.swap_tmore(t_indexes, c_index, *default_parameters)


def apply_reverse_matrix(
    gate: BasicGateTensor, default_parameters, algorithm, n_qubits
):
    # Get gate's parameters
    assert gate.matrix_type == MatrixType.reverse
    args_num = gate.controls + gate.targets
    gate_args = gate.cargs + gate.targs
    matrix = cp.from_dlpack(gate.matrix.clone())

    if args_num == 1:  # Deal with 1-qubit reverse gate, e.g. Y
        index = n_qubits - 1 - gate_args[0]
        algorithm.reverse_targ(index, matrix, *default_parameters)
    elif args_num == 2:  # only consider 1 control qubit + 1 target qubit
        c_index = n_qubits - 1 - gate_args[0]
        t_index = n_qubits - 1 - gate_args[1]
        algorithm.reverse_ctargs(c_index, t_index, matrix, *default_parameters)
    else:  # CCX
        c_indexes = [n_qubits - 1 - carg for carg in gate.cargs]
        t_index = n_qubits - 1 - gate.targ
        algorithm.reverse_more(c_indexes, t_index, *default_parameters)


def apply_control_matrix(
    gate: BasicGateTensor, default_parameters, algorithm, n_qubits
):
    # Get gate's parameters
    assert gate.matrix_type == MatrixType.control
    args_num = gate.controls + gate.targets
    gate_args = gate.cargs + gate.targs

    if args_num == 1:  # Deal with 1-qubit control gate, e.g. S
        index = n_qubits - 1 - gate_args[0]
        val = gate.matrix[1, 1]
        algorithm.control_targ(index, val, *default_parameters)
    elif args_num == 2:  # Deal with 2-qubit control gate, e.g. CZ
        c_index = n_qubits - 1 - gate_args[0]
        t_index = n_qubits - 1 - gate_args[1]
        val = gate.matrix[3, 3]
        algorithm.control_ctargs(c_index, t_index, val, *default_parameters)


class Applygate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state_in, gate, requires_grad, fp, algorithm, n_qubits):
        ctx.gate = gate
        ctx.algorithm = algorithm
        ctx.n_qubits = n_qubits
        cupy_state = cp.from_dlpack(state_in.clone())
        default_parameters = (cupy_state, n_qubits, True)

        # Deal with quantum gate with more than 3 qubits.
        if (
            gate.type in [GateType.unitary, GateType.qft, GateType.iqft]
            and gate.targets >= 3
        ):
            matrix = cp.from_dlpack(gate.matrix.clone())
            matrix = matrix.reshape(
                1 << (gate.controls + gate.targets), 1 << (gate.controls + gate.targets)
            )
            cupy_state = algorithm.matrix_dot_vector(
                cupy_state, n_qubits, matrix, gate.cargs + gate.targs, True
            )
            state_out = torch.from_dlpack(cupy_state)
            ctx.save_for_backward(state_in, requires_grad)
            return state_out

        matrix_type = gate.matrix_type
        # [H, SX, SY, SW, U2, U3, Rx, Ry] 2-bits [CH, ] 2-bits[targets] [unitary]
        if matrix_type == MatrixType.normal:
            apply_normal_matrix(gate, default_parameters, algorithm, n_qubits, fp)
        # [Rz, Phase], 2-bits [CRz, Rzz], 3-bits [CCRz]
        elif matrix_type in [MatrixType.diagonal, MatrixType.diag_diag]:
            apply_diagonal_matrix(gate, default_parameters, algorithm, n_qubits, fp)
        # [X] 2-bits [swap] 3-bits [CSWAP]
        elif matrix_type == MatrixType.swap:
            assert fp is True
            apply_swap_matrix(gate, default_parameters, algorithm, n_qubits)
        # [Y] 2-bits [CX, CY] 3-bits: [CCX]
        elif matrix_type == MatrixType.reverse:
            assert fp is True
            apply_reverse_matrix(gate, default_parameters, algorithm, n_qubits)
        # [S, sdg, Z, U1, T, tdg] # 2-bits [CZ, CU1]
        elif matrix_type == MatrixType.control:
            assert fp is True
            apply_control_matrix(gate, default_parameters, algorithm, n_qubits)
        # [Rxx, Ryy]
        elif matrix_type == MatrixType.normal_normal:
            apply_normal_normal_matrix(
                gate, default_parameters, algorithm, n_qubits, fp
            )
        # [Rzx]
        elif matrix_type == MatrixType.diag_normal:
            apply_diagonal_normal_matrix(
                gate, default_parameters, algorithm, n_qubits, fp
            )

        state_out = torch.from_dlpack(cupy_state)
        ctx.save_for_backward(state_in, requires_grad)
        return state_out

    @staticmethod
    def backward(ctx, grad_output):
        state_in, requires_grad = ctx.saved_tensors
        if not requires_grad:
            return None
        grad_parg = Applygate().apply(
            state_in, ctx.gate, requires_grad, False, ctx.algorithm, ctx.n_qubits
        )
        return None, None, grad_parg.real, None, None, None


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.algorithm = LinAlgLoader(
            device="GPU", enable_gate_kernel=True, enable_multigpu_gate_kernel=False,
        )
        self.n_qubit = 2
        self.device = torch.device("cuda:0")
        self.params = torch.nn.Parameter(
            torch.rand(1, device=self.device), requires_grad=True,
        )

    def forward(self, state):
        gate = Rx_tensor(self.params)
        print(gate.pargs.requires_grad)
        gate.targs = [0]
        state = Applygate.apply(
            state, gate, self.params, True, self.algorithm, self.n_qubit
        )

        return state


def seed(seed: int):
    """Set random seed.

        Args:
            seed (int): The random seed.
        """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    seed(0)
    n_qubits = 2
    device = torch.device("cuda:0")
    state = torch.zeros(1 << n_qubits, dtype=torch.complex128).to(device)
    state[0] = 1

    net = Net()
    optimizer = torch.optim.Adam
    optim = optimizer([dict(params=net.parameters(), lr=0.1)])

    optim.zero_grad()
    state = net(state)
    # # print("state1", state)
    # loss = sum(state).real
    # loss.backward()
    # print(net.params.grad)
    # optim.step()

    # # test ansatz
    # ansatz = Ansatz(n_qubits, device=torch.device("cpu"))
    # params = torch.nn.Parameter(
    #     torch.tensor([0.3990]).to(torch.device("cpu")), requires_grad=True,
    # )
    # optim = optimizer([dict(params=params, lr=0.1)])
    # ansatz.add_gate(Rx_tensor(params), 0)
    # optim.zero_grad()
    # state, _ = ansatz.forward()
    # # print("state2", state)
    # loss = sum(state).real
    # loss.backward()
    # print(params.grad)
    # optim.step()
