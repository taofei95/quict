import os
import random
import shutil
import time
from typing import Union

import numpy as np
import torch
import tqdm

from QuICT.algorithm.quantum_machine_learning.utils import Hamiltonian
from QuICT.algorithm.quantum_machine_learning.VQA.model import QAOANet

OPTIMIZER_LIST = [
    "Adadelta",
    "Adagrad",
    "Adam",
    "AdamW",
    "SparseAdam",
    "Adamax",
    "ASGD",
    "LBFGS",
    "NAdam",
    "RAdam",
    "RMSprop",
    "Rprop",
    "SGD",
]


class QAOA:
    """Quantum Approximate Optimization Algorithm. User interface class.

    QAOA <https://arxiv.org/abs/1411.4028> is a algorithm for finding approximate
    solutions to combinatorial-optimization problems.
    """

    def __init__(
        self,
        n_qubits: int,
        p: int,
        hamiltonian: Hamiltonian,
        loss_func=None,
        seed: int = 0,
        device="cuda:0",
    ):
        """Complete QAOA algorithm process instance.

        Args:
            n_qubits (int): The number of qubits.
            p (int): The number of QAOA layers.
            hamiltonian (Hamiltonian): The hamiltonian for a specific task.
            loss_func (optional): The User-defined loss function.
                Defaults to None, which means using the build-in default loss function.
            seed (int, optional): The random seed. Defaults to 0.
            device (str, optional): The device to which the model is assigned.
                Defaults to "cuda:0".
        """
        self.n_qubits = n_qubits
        self.device = torch.device(device)
        self._seed(seed)
        self.net = QAOANet(n_qubits, p, hamiltonian, self.device)
        self.loss_func = loss_func if loss_func is not None else self.net.loss_func
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

    def _save_checkpoint(self, it, latest=False):
        """Save the model and optimizer as a checkpoint.

        Args:
            it (int): The number of the saved iteration.
            latest (bool, optional): Whether this is the last iteration. Defaults to False.
        """
        os.makedirs(self.model_path, exist_ok=True)
        checkpoint = dict(
            iter=it, graph=self.net.state_dict(), optimizer=self.optim.state_dict()
        )
        torch.save(checkpoint, "{0}/model.ckpt".format(self.model_path))
        if not latest:
            shutil.copy(
                "{0}/model.ckpt".format(self.model_path),
                "{0}/{1}.ckpt".format(self.model_path, it),
            )

    def _restore_checkpoint(self, resume):
        """Restore the model and optimizer from a checkpoint.

        Args:
            resume (int/bool): If resume is True, restore the latest checkpoint.
                Or users can specify a checkpoint saved in an iteration to restore.

        Returns:
            int: The number of the restored iteration.
        """
        assert resume and self.model_path
        try:
            model_name = (
                "{0}/model.ckpt".format(self.model_path)
                if resume is True
                else "{0}/{1}.ckpt".format(self.model_path, resume)
            )
        except:
            raise Exception("Cannot find the model.")

        checkpoint = torch.load(model_name, map_location=self.device)
        try:
            self.net.load_state_dict(checkpoint["graph"])
            if self.optim:
                self.optim.load_state_dict(checkpoint["optimizer"])
        except:
            raise Exception("Cannot load the model correctly.")

        it = checkpoint["iter"]
        if resume is not True:
            assert resume == it

        return it

    def train(
        self,
        optimizer: str,
        lr: float,
        max_iters: int,
        save_model=True,
        model_path=None,
        ckp_freq: int = 10,
        resume: Union[bool, int] = False,
    ):
        """The training process.

        Args:
            optimizer (str): The built-in optimizers in Pytorch. Need to choose from OPTIMIZER_LIST.
            lr (float): The learning rate.
            max_iters (int): The maximum number of iterations to train.
            save_model (bool, optional): Whether to save the models. Defaults to True.
            model_path (str, optional): The specified model save path.
                Defaults to None, and a new folder will be created based on the timestamp.
            ckp_freq (int, optional): The number of iterations to interval to save a checkpoint. Defaults to 10.
            resume (Union[bool, int], optional): Whether to restore an existing model and continue training.
                Defaults to False. If False, train from scratch. If True, restore the latest checkpoint.
                Or users can specify a checkpoint saved in an iteration to restore.

        Returns:
            torch.Tensor: The final state vector.
        """
        now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
        self.model_path = (
            "QAOA_model_" + now if save_model and model_path is None else model_path
        )
        assert optimizer in OPTIMIZER_LIST
        optimizer = getattr(torch.optim, optimizer)
        self.optim = optimizer([dict(params=self.net.parameters(), lr=lr)])

        # Restore checkpoint
        curr_it = (
            self._restore_checkpoint(resume=resume) if resume and model_path else 0
        )
        assert curr_it < max_iters

        # Start training
        self.net.train()
        loader = tqdm.trange(curr_it, max_iters, desc="Training", leave=False)
        for it in loader:
            self.optim.zero_grad()
            state = self.net()
            loss = self.loss_func(state)
            loss.backward()
            self.optim.step()
            loader.set_postfix(loss=loss.item())

            # Save checkpoint
            if save_model and ((it + 1) % ckp_freq == 0 or (it + 1) == max_iters):
                self._save_checkpoint(it + 1, latest=(it + 1 == max_iters))

        return state

    def test(self, model_path):
        """The testing process.

        Args:
            model_path (str): The save path of the model to be tested.

        Returns:
            torch.Tensor: The final state vector.
        """
        assert model_path is not None and model_path != ""
        self.model_path = model_path
        # Restore checkpoint
        self._restore_checkpoint(resume=True)
        self.net.eval()
        state = self.net()
        return state
