import numpy as np
import torch
import torch.utils.data as data
import tqdm
import time
from typing import Union
import torch.utils.tensorboard

from QuICT.algorithm.quantum_machine_learning.dataset import MNISTDataset
from QuICT.algorithm.quantum_machine_learning.model.QNN import QuantumNet
from QuICT.algorithm.quantum_machine_learning.utils.ml_utils import *


class QNNMnistClassifier:
    """Quantum Neural Network for MNIST classification

    Google Tensorflow Quantum: https://arxiv.org/abs/1802.06002
    https://www.tensorflow.org/quantum/tutorials/mnist
    """

    def __init__(
        self,
        class0=3,
        class1=6,
        resize=(4, 4),
        threshold=0.5,
        encoding: str = "qubit",
        layers=["XX", "ZZ"],
        loss_func=None,
        seed: int = 0,
        device="cuda:0",
    ):
        """The complete algorithm process of using QNN to classify MNIST handwritten dataset.

        Args:
            class0 (int, optional): Class marked as 0. Defaults to 3.
            class1 (int, optional): Class marked as 1. Defaults to 6.
            resize (tuple, optional): The size of the downscaled images for classification. Defaults to (4, 4).
            threshold (float, optional): Threshold for image binarization.
                Pixels greater than threshold are regarded as 1, otherwise are regarded as 0. Defaults to 0.5.
            encoding (str, optional): The encoding method to encode the image as quantum ansatz.
                Only support qubit encoding and amplitude encoding. Defaults to "qubit".
            layers (list, optional): The list of types of QNN layers.
                Currently only supports XX, YY, ZZ, and ZX. Defaults to ["XX", "ZZ"].
            loss_func (optional): The User-defined loss function.
                Defaults to None, which means using the build-in default loss function.
            seed (int, optional): The random seed. Defaults to 0.
            device (str, optional): The device to which the model is assigned.
                Defaults to "cuda:0".
        """
        set_seed(seed)
        self.class0 = class0
        self.class1 = class1
        self.resize = resize
        self.threshold = threshold
        self.encoding = encoding
        self.device = torch.device(device)

        self.net = QuantumNet(resize[0] * resize[1], layers, encoding, self.device)
        self.loss_func = loss_func if loss_func is not None else self._hinge_loss
        self.model_path = None
        self.optim = None

    def _hinge_loss(self, y_true, y_pred):
        """Calculate the hinge loss and record the number of correct classifications."""
        y_true = 2 * y_true.type(torch.float32) - 1.0
        y_pred = 2 * y_pred - 1.0
        loss = torch.clamp(1 - y_pred * y_true, min=0.0)
        correct = torch.where(y_true * y_pred > 0)[0].shape[0]
        return torch.mean(loss), correct

    def _load_data(self, batch_size=1):
        """Load data and perform data preprocessing."""
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
        lr: float,
        epoch: int = 3,
        batch_size: int = 32,
        save_model=True,
        model_path=None,
        ckp_freq: int = 30,
        resume: Union[bool, dict] = False,
    ):
        """The training process.

        Args:
            lr (float): The learning rate.
            epoch (int, optional): The number of epoch to train. Defaults to 3.
            batch_size (int, optional): The number of samples that will be passed through to the network at one time. Defaults to 32.
            save_model (bool, optional): Whether to save the models. Defaults to True.
            model_path (str, optional): The specified model save path.
                Defaults to None, and a new folder will be created based on the timestamp.
            ckp_freq (int, optional): The number of iterations to interval to save a checkpoint. Defaults to 30.
            resume (Union[bool, dict], optional): Whether to restore an existing model and continue training.
                Defaults to False. If False, train from scratch. If True, restore the latest checkpoint.
                Or users can specify a checkpoint saved in an iteration to restore. eg. {"ep": 0, "it": 300}.
        """
        now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
        # Set model path and initialize tensorboard
        if save_model and model_path is None:
            self.model_path = "QNN_MNIST_%d%d_" % (self.class0, self.class1) + now
            tb = torch.utils.tensorboard.SummaryWriter(
                log_dir=self.model_path + "/logs"
            )
        else:
            self.model_path = model_path
            tb = None
        self.optim = torch.optim.Adam([dict(params=self.net.parameters(), lr=lr)])
        # Load MNIST dataset and preprocessing the data.
        self._load_data(batch_size)

        # Restore checkpoint
        curr_ep, curr_it = (
            restore_checkpoint(
                self.net, self.optim, self.model_path, self.device, resume
            )
            if resume and model_path
            else (0, 0)
        )
        assert curr_ep < epoch

        # train epoch
        for ep in range(curr_ep, epoch):
            self.net.train()
            loader = tqdm.tqdm(
                self.train_loader, desc="Training epoch {}".format(ep + 1), leave=True
            )
            # train iteration
            assert curr_it < len(loader)
            for it, (x_train, y_train) in enumerate(loader, start=curr_it):
                self.optim.zero_grad()
                y_pred = self.net(x_train)
                loss, correct = self.loss_func(y_train, y_pred)
                accuracy = correct / batch_size
                loss.backward()
                self.optim.step()
                loader.set_postfix(
                    it=it,
                    loss="{:.3f}".format(loss),
                    accuracy="{:.3f}".format(accuracy),
                )
                # Tensorboard
                if tb:
                    tb.add_scalar("train/loss", loss, ep * len(loader) + it)
                    tb.add_scalar("train/accuracy", accuracy, ep * len(loader) + it)

                # Save checkpoint
                latest = (ep + 1) == epoch and (it + 1) == len(loader)
                if save_model and ((it + 1) % ckp_freq == 0 or latest):
                    save_checkpoint(
                        self.net, self.optim, self.model_path, ep, it + 1, latest
                    )

            # Validation
            self.net.eval()
            loader_val = tqdm.tqdm(
                self.test_loader, desc="Validating epoch {}".format(ep + 1), leave=True
            )
            loss_val_list = []
            total_correct = 0
            for it, (x_test, y_test) in enumerate(loader_val):
                y_pred = self.net(x_test)
                loss_val, correct = self.loss_func(y_test, y_pred)
                loss_val_list.append(loss_val.cpu().detach().numpy())
                total_correct += correct
                accuracy_val = correct / batch_size
                loader_val.set_postfix(
                    it=it,
                    loss="{:.3f}".format(loss_val),
                    accuracy="{:.3f}".format(accuracy_val),
                )
            if tb:
                tb.add_scalar("validation/loss", np.mean(loss_val_list), ep)
                tb.add_scalar(
                    "validation/accuracy",
                    total_correct / (len(loader_val) * batch_size),
                    ep,
                )

    def test(self, model_path):
        """The testing process.

        Args:
            model_path (str): The save path of the model to be tested.

        Returns:
            float: Classification accuracy.
        """
        assert model_path is not None and model_path != ""
        # Load MNIST dataset and preprocessing the data.
        self._load_data()
        # Restore checkpoint
        restore_checkpoint(self.net, None, model_path, self.device, resume=True)
        self.net.eval()
        loader = tqdm.tqdm(self.test_loader, desc="Testing", leave=True)
        total_correct = 0
        for it, (x_test, y_test) in enumerate(loader):
            y_pred = self.net(x_test)
            loss, correct = self.loss_func(y_test, y_pred)
            total_correct += correct
            loader.set_postfix(
                it=it,
                loss="{:.3f}".format(loss),
                correct="{0}".format(bool(correct)),
            )

        accuracy = total_correct / len(loader)
        return accuracy


c = QNNMnistClassifier(threshold=0.5, resize=(4, 4), device="cuda:0", seed=17)
c.train(lr=0.001, batch_size=32)
