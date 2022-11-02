import torch

from QuICT.algorithm.quantum_machine_learning.ansatz_library import QNN_Layer
from QuICT.algorithm.quantum_machine_learning.utils import Ansatz
from QuICT.algorithm.quantum_machine_learning.utils.gate_tensor import *
from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.simulation.state_vector import ConstantStateVectorSimulator


"""Google Tensorflow Quantum https://arxiv.org/abs/1802.06002
"""





import tqdm

c = MNIST_Classifier(threshold=0.1)
optimizer = torch.optim.Adam
optim = optimizer([dict(params=c.pqc.parameters(), lr=0.1)])

model = torch.nn.Sequential(torch.nn.Linear(32, 32))


# Start training
c.pqc.train()
loader = tqdm.trange(0, 100, desc="Training", leave=False)
for it in loader:
    optim.zero_grad()
    state = c.forward()
    loss = sum(state)
    loss.backward()
    optim.step()
    loader.set_postfix(loss=loss.item())

