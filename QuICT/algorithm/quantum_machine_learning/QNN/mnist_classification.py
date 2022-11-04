import numpy as np
import random
import torch
import tqdm

from QuICT.algorithm.quantum_machine_learning.dataset import MNISTDataset
from QuICT.algorithm.quantum_machine_learning.QNN.model import QuantumNet, ClassicalNet
from QuICT.simulation.state_vector import ConstantStateVectorSimulator


"""Google Tensorflow Quantum https://arxiv.org/abs/1802.06002
"""


class QNNMnistClassifier:
    def __init__(
        self,
        class0=3,
        class1=6,
        resize=(4, 4),
        threshold=0.1,
        layers=["XX", "ZZ"],
        loss_func=None,
        seed: int = 0,
        device="cuda:0",
    ):
        self.class0 = class0
        self.class1 = class1
        self.resize = resize
        self.threshold = threshold
        self.device = torch.device(device)

        self._seed(seed)
        self.net = QuantumNet(resize[0] * resize[1], layers, self.device)
        self.loss_func = loss_func if loss_func is not None else self.loss_func
        self.model_path = None
        self.optim = None

    def _seed(self, seed: int):
        """Set random seed.

        Args:
            seed (int): The random seed.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def loss_func(self):
        return

    def train(self):
        # Load MNIST dataset and preprocessing the data.
        dataset = MNISTDataset(device=self.device)
        dataset.filter_targets(self.class0, self.class1)
        dataset.downscale(self.resize)
        dataset.remove_conflict(self.resize)
        dataset.binary_img(self.threshold)


c = QNNMnistClassifier(threshold=0.1)
c.train()
# optimizer = torch.optim.Adam
# optim = optimizer([dict(params=c.pqc.parameters(), lr=0.1)])

# model = torch.nn.Sequential(torch.nn.Linear(32, 32))


# # Start training
# c.pqc.train()
# loader = tqdm.trange(0, 100, desc="Training", leave=False)
# for it in loader:
#     optim.zero_grad()
#     state = c.forward()
#     loss = sum(state)
#     loss.backward()
#     optim.step()
#     loader.set_postfix(loss=loss.item())

