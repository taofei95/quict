import torch
import numpy as np
import time

# device = torch.device("cuda:0")
# dim = 65536
def f(state):
    H = torch.tensor(
        [[1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), -1 / np.sqrt(2)]],
        dtype=torch.complex128,
    ).to(device)
    I = torch.eye(2, dtype=torch.complex128).to(device)
    H_matrix = torch.tensor([[1]], dtype=torch.complex128,).to(device)
    for i in range(n_qubits):
        H_matrix = torch.kron(H_matrix, H)
    
    return torch.mm(state, H_matrix)


n_qubits = 16
device = torch.device("cpu")
state = torch.zeros(1, 1 << n_qubits, dtype=torch.complex128).to(device)
state[0, 0] = 1
s = time.time()
state = f(state)
print(time.time() - s)
