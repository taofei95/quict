import time
from typing import Union

import torch
import tqdm

from QuICT.algorithm.quantum_machine_learning.utils import Hamiltonian
from QuICT.algorithm.quantum_machine_learning.utils.ml_utils import *
from QuICT.algorithm.quantum_machine_learning.VQA.model import QAOANet


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
        set_seed(seed)
        self.n_qubits = n_qubits
        self.device = torch.device(device)
        self.net = QAOANet(n_qubits, p, hamiltonian, self.device)
        self.loss_func = loss_func if loss_func is not None else self.net.loss_func
        self.model_path = None
        self.optim = None

    def train(
        self,
        optimizer: str,
        lr: float,
        max_iters: int,
        save_model=True,
        model_path=None,
        ckp_freq: int = 10,
        resume: Union[bool, dict] = False,
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
            resume (Union[bool, dict], optional): Whether to restore an existing model and continue training.
                Defaults to False. If False, train from scratch. If True, restore the latest checkpoint.
                Or users can specify a checkpoint saved in an iteration to restore. eg. {"ep": 0, "it": 300}

        Returns:
            torch.Tensor: The final state vector.
        """
        now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
        self.model_path = (
            "QAOA_model_" + now if save_model and model_path is None else model_path
        )
        self.optim = set_optimizer(optimizer, self.net, lr)

        # Restore checkpoint
        curr_it = (
            restore_checkpoint(
                self.net, self.optim, self.model_path, self.device, resume
            )[1]
            if resume and model_path
            else 0
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
            latest = it + 1 == max_iters
            if save_model and ((it + 1) % ckp_freq == 0 or latest):
                save_checkpoint(
                    self.net, self.optim, self.model_path, 0, it + 1, latest=latest
                )

        return state

    def test(self, model_path):
        """The testing process.

        Args:
            model_path (str): The save path of the model to be tested.

        Returns:
            torch.Tensor: The final state vector.
        """
        assert model_path is not None and model_path != ""
        # Restore checkpoint
        restore_checkpoint(self.net, None, model_path, self.device, resume=True)
        self.net.eval()
        state = self.net()
        return state
