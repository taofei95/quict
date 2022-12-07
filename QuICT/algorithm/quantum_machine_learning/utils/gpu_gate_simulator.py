import cupy as cp
import torch
import random
import time


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
    assert gate.matrix_type == MatrixType.diag_normal
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
    matrix = cp.from_dlpack(gate.matrix.clone()).get()

    if args_num == 1:  # Deal with 1-qubit control gate, e.g. S
        index = n_qubits - 1 - gate_args[0]
        val = matrix[1, 1]
        algorithm.control_targ(index, val, *default_parameters)
    elif args_num == 2:  # Deal with 2-qubit control gate, e.g. CZ
        c_index = n_qubits - 1 - gate_args[0]
        t_index = n_qubits - 1 - gate_args[1]
        val = matrix[3, 3]
        algorithm.control_ctargs(c_index, t_index, val, *default_parameters)


def apply_gate(gate, default_parameters, algorithm, fp):
    cupy_state = default_parameters[0]
    n_qubits = default_parameters[1]
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
        apply_normal_normal_matrix(gate, default_parameters, algorithm, n_qubits, fp)
    # [Rzx]
    elif matrix_type == MatrixType.diag_normal:
        apply_diagonal_normal_matrix(gate, default_parameters, algorithm, n_qubits, fp)
    # [Measure, Reset]
    # elif gate.type in [GateType.measure, GateType.reset]:
    #     index = n_qubits - 1 - gate.targ
    #     prob = self.get_measured_prob(index).get()
    #     self.apply_specialgate(index, gate_type, prob)

    state_out = torch.from_dlpack(cupy_state)
    return state_out


class Applygate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, ansatz, pargs, algorithm, n_qubits):
        pargs = pargs.flatten()
        pargs_ptr = ansatz.trainable_pargs_ptr
        grads = [None] * pargs.shape[0]

        for gate in ansatz.gates:
            if gate.type == GateType.measure:
                raise ValueError
            cupy_state = cp.from_dlpack(state.clone())
            default_parameters = (cupy_state, n_qubits, True)
            state_out = apply_gate(gate, default_parameters, algorithm, True)

            for i in range(len(grads)):
                idx = (
                    pargs_ptr.index(gate.pargs.data_ptr())
                    if gate.pargs.requires_grad
                    else None
                )
                # fx = A * B: grad = 0 [TESTED]
                if i != idx and grads[i] is None:
                    continue

                # fx = Gate * state(x): grad = Gate * grad_pre [TESTING]
                elif i != idx and grads[i] is not None:
                    cupy_grad = cp.from_dlpack(grads[i].conj().clone())
                    default_parameters = (cupy_grad, n_qubits, True)
                    grads[i] = apply_gate(
                        gate, default_parameters, algorithm, True
                    ).conj()

                # fx = Gate(x) * state: grad = Gate'(x) * state [TESTING]
                elif i == idx and grads[i] is None:
                    cupy_state = cp.from_dlpack(state.clone())
                    default_parameters = (cupy_state, n_qubits, True)
                    grads[idx] = apply_gate(
                        gate, default_parameters, algorithm, False
                    ).conj()

                # fx = Gate(x) * state(x): grad = Gate'(x) * state(x) + Gate(x) * grad_pre [TESTING]
                else:
                    cupy_state = cp.from_dlpack(state.clone())
                    default_parameters = (cupy_state, n_qubits, True)
                    grad1 = apply_gate(
                        gate, default_parameters, algorithm, False
                    ).conj()
                    cupy_grad = cp.from_dlpack(grads[idx].conj().clone())
                    default_parameters = (cupy_grad, n_qubits, True)
                    grad2 = apply_gate(gate, default_parameters, algorithm, True).conj()
                    grads[idx] = grad1 + grad2
            state = state_out

        # Turn None to zeros
        for i in range(len(grads)):
            if grads[i] is None:
                grads[i] = torch.zeros(1 << n_qubits, dtype=state.dtype).to(
                    state.device
                )

        grads = torch.stack(grads)
        ctx.save_for_backward(grads)
        return state

    @staticmethod
    def backward(ctx, grad_output):
        (grad,) = ctx.saved_tensors
        grad = grad_output * grad
        return None, None, grad.real, None, None, None


def gpu_forward(state, ansatz, n_qubits):
    algorithm = LinAlgLoader(
        device="GPU",
        enable_gate_kernel=True,
        enable_multigpu_gate_kernel=False,
    )

    if state is None:
        state = torch.zeros(1 << n_qubits, dtype=torch.complex128).to(ansatz.device)
        state[0] = 1
    else:
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(ansatz.device)
        else:
            state = state.clone()
    assert state.shape[0] == 1 << n_qubits

    params = torch.stack(ansatz.trainable_pargs)
    state = Applygate.apply(state, ansatz, params, algorithm, n_qubits)
    return state, None


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.n_qubit = 4
        self.device = torch.device("cuda:0")
        self.params = torch.nn.Parameter(
            torch.tensor([[-0.1456], [0.1234]]).to(self.device),
            requires_grad=True,
        )

    def forward(self):
        ansatz_gpu = Ansatz(self.n_qubit)
        ansatz_gpu.add_gate(H_tensor)
        ansatz_gpu.add_gate(Rz_tensor(0.333), 0)
        ansatz_gpu.add_gate(Rx_tensor(self.params[1]), 0)
        ansatz_gpu.add_gate(Ry_tensor(0.211), 1)
        ansatz_gpu.add_gate(Rz_tensor(self.params[1]), 0)
        ansatz_gpu.add_gate(Rzx_tensor(-self.params[0]), [1, 3])
        ansatz_gpu.add_gate(Rxx_tensor(self.params[1]), [1, 0])

        state = None
        state, _ = gpu_forward(state, ansatz_gpu, self.n_qubit)
        return state


class CPUNet(torch.nn.Module):
    def __init__(self):
        super(CPUNet, self).__init__()
        self.n_qubit = 4
        self.device = torch.device("cpu")
        self.params = torch.nn.Parameter(
            torch.tensor([[-0.1456], [0.1234]]).to(self.device),
            requires_grad=True,
        )

    def forward(self):
        ansatz_cpu = Ansatz(self.n_qubit, device=self.device)
        ansatz_cpu.add_gate(H_tensor)
        ansatz_cpu.add_gate(Rz_tensor(0.333), 0)
        ansatz_cpu.add_gate(Rz_tensor(self.params[1]), 0)
        ansatz_cpu.add_gate(Rz_tensor(0.211), 1)
        ansatz_cpu.add_gate(Rz_tensor(self.params[1]), 0)

        state, _ = ansatz_cpu.forward()
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
    from QuICT.algorithm.quantum_machine_learning.utils import Ansatz

    seed(0)
    device = torch.device("cuda:0")

    net = Net()
    optimizer = torch.optim.SGD
    optim = optimizer([dict(params=net.parameters(), lr=0.01)])
    net.train()

    cpu_net = CPUNet()
    optim_cpu = optimizer([dict(params=cpu_net.parameters(), lr=0.01)])
    cpu_net.train()
    aim_state = torch.tensor(
        [
            0.25,
            0.25,
            0.25,
            0.25,
            0.25,
            0.25,
            0.25,
            0.25,
            0.25,
            0.25,
            0.25,
            0.25,
            0.25,
            0.25,
            0.25,
            0.25,
        ],
        dtype=torch.complex128,
    )

    for i in range(1):
        print("Iteration", i)
        optim.zero_grad()
        optim_cpu.zero_grad()
        state = net()
        state_cpu = cpu_net()
        print(torch.dist(state.cpu(), state_cpu))
        print("-----------------------------")
        loss = torch.dist(state, aim_state.to(device), p=2)
        loss_cpu = torch.dist(state_cpu, aim_state, p=2)
        loss.backward()
        loss_cpu.backward()
        print("grad - gpu", net.params.grad)
        print("grad - cpu", cpu_net.params.grad)
        print("-----------------------------")
        optim.step()
        optim_cpu.step()
