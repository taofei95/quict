import torch
import tqdm

from QuICT.algorithm.quantum_machine_learning.QNN.model import QuantumNet, ClassicalNet
from QuICT.simulation.state_vector import ConstantStateVectorSimulator


"""Google Tensorflow Quantum https://arxiv.org/abs/1802.06002
"""


class QNNMnistClassifier:
    def __init__(
        self, layers=["XX", "ZZ"], loss_func=None, seed: int = 0, device="cuda:0",
    ):
        self.device = torch.device(device)
        self._seed(seed)
        self.net = QuantumNet(layers, self.device)
        self.loss_func = loss_func if loss_func is not None else self.loss_func
        self.model_path = None
        self.optim = None


c = QNNMnistClassifier(threshold=0.1)
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

