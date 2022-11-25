import numpy as np
import random
import torch
import torch.utils.data as data
import tqdm
import time
from typing import Union

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
        encoding: str = "qubit",
        layers=["XX", "ZZ"],
        loss_func=None,
        seed: int = 0,
        device="cuda:0",
    ):
        self.class0 = class0
        self.class1 = class1
        self.resize = resize
        self.threshold = threshold
        self.encoding = encoding
        self.device = torch.device(device)

        self._seed(seed)
        self.net = QuantumNet(resize[0] * resize[1], layers, encoding, self.device)
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

    def loss_func(self, y_true, y_pred):
        y_true = 2 * y_true - 1.0
        y_pred = 2 * y_pred - 1.0
        loss = torch.clamp(1 - y_pred * y_true, min=0.0)
        return torch.mean(loss)

    def train_batch(self, batch_size):
        return

    def _load_data(self, batch_size):
        train_data = MNISTDataset(train=True, device=self.device)
        train_data.filter_targets(self.class0, self.class1)
        train_data.downscale(self.resize)
        train_data.remove_conflict(self.resize)
        train_data.binary_img(self.threshold)

        test_data = MNISTDataset(train=False, device=self.device)
        test_data.filter_targets(self.class0, self.class1)
        test_data.downscale(self.resize)
        test_data.remove_conflict(self.resize)
        test_data.binary_img(self.threshold)

        self.train_loader = data.DataLoader(
            dataset=train_data, batch_size=batch_size, shuffle=True
        )
        self.test_loader = data.DataLoader(
            dataset=test_data, batch_size=batch_size, shuffle=True
        )

    def train(
        self,
        optimizer: str,
        lr: float,
        epoch: int = 3,
        batch_size: int = 32,
        save_model=True,
        model_path=None,
        ckp_freq: int = 10,
        resume: Union[bool, int] = False,
    ):
        # Load MNIST dataset and preprocessing the data.
        self._load_data(batch_size)

        optimizer = getattr(torch.optim, optimizer)
        self.optim = optimizer([dict(params=self.net.parameters(), lr=lr)])
        it = 0
        # Start training
        self.net.train()
        # train epoch
        for ep in range(0, epoch):
            loader = tqdm.tqdm(
                self.train_loader, desc="training epoch {}".format(ep + 1), leave=False
            )
            # train iteration
            for batch in loader:
                self.optim.zero_grad()
                x_train = batch[0]
                y_train = batch[1]
                y_pred = self.net(x_train)
                loss = self.loss_func(y_train, y_pred)
                loss.backward()
                for para in self.net.parameters():
                    print(para.grad)
                self.optim.step()
                it += 1
                it_end = time.time()
                loader.set_postfix(it=it, loss="{:.3f}".format(loss))

        # # Save checkpoint
        # if save_model and ((it + 1) % ckp_freq == 0 or (it + 1) == max_iters):
        #     self._save_checkpoint(it + 1, latest=(it + 1 == max_iters))


c = QNNMnistClassifier(threshold=0.1, resize=(2, 2), device="cpu")
c.train(optimizer="Adam", lr=0.1, batch_size=1)
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



