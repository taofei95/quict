import numpy as np
import QuICT.ops.linalg.cpu_calculator as CPUCalculator


def linear_algorithm():
    A = np.random.random((1 << 3, 1 << 3)).astype(np.complex128)
    B = np.random.random((1 << 3, 1 << 3)).astype(np.complex128)

    np_tensor = np.kron(A, B)
    cpu_result = CPUCalculator.tensor(A, B)
    print(np.allclose(np_tensor, cpu_result))

    # Shuffle State Vector with random qubits order.
    random_sv = np.random.random(1 << 5).astype(np.complex128)
    mapping = np.array(list(range(5)))
    np.random.shuffle(mapping)

    shuffle_sv = CPUCalculator.VectorPermutation(random_sv, mapping, changeInput=False)


def lalg_GPU():
    # Need GPU environment Support.
    import QuICT.ops.linalg.gpu_calculator as GPUCalculator

    A = np.random.random((1 << 3, 1 << 3)).astype(np.complex128)
    B = np.random.random((1 << 3, 1 << 3)).astype(np.complex128)

    np_tensor = np.kron(A, B)
    gpu_result = GPUCalculator.tensor(A, B, gpu_out=True)
    print(np.allclose(np_tensor, gpu_result))


if __name__ == "__main__":
    linear_algorithm()
