import os
import random
import shutil

import numpy as np

from QuICT.algorithm.quantum_machine_learning.optimizer.initializer import (
    OptimizerInitializer,
)
from QuICT.core.utils import Variable
from QuICT.tools.exception.algorithm import *
from QuICT.tools.logger import *

logger = Logger("ML_utils")


def set_seed(seed: int):
    """Set random seed.

    Args:
        seed (int): The random seed.
    """
    np.random.seed(seed)
    random.seed(seed)


def save_checkpoint(model, model_path: str, ep: int, it: int, latest=False):
    """Save the model and optimizer as a checkpoint.

    Args:
        model (Model): The network that to be saved.
        model_path (str): The path to the saved checkpoint.
        ep (int): The number of the saved epoch.
        it (int): The number of the saved iteration.
        latest (bool, optional): Whether this is the last iteration. Defaults to False.
    """
    os.makedirs(model_path, exist_ok=True)
    circuit_dict = dict(
        pargs=model.params.pargs,
        grads=model.params.grads,
        identity=model.params.identity,
        shape=model.params.shape,
    )
    optim = model.optimizer
    optim_dict = dict(cache=optim.cache, hyperparameters=optim.hyperparameters)
    checkpoint = dict(epoch=ep, iter=it, circuit=circuit_dict, optimizer=optim_dict)
    np.save("{0}/model.npy".format(model_path), checkpoint)

    if not latest:
        shutil.copy(
            "{0}/model.npy".format(model_path),
            "{0}/{1}_{2}.npy".format(model_path, ep, it),
        )


def restore_checkpoint(model, model_path, restore_optim=True):
    """Restore the model and optimizer from a checkpoint.

    Args:
        net (Model): The network that to be restored.
        model_path (str): The path to the saved checkpoint.
        restore_optim (bool, optional):  Whether to restore optimizer. Defaults to True.

    Returns:
        int: The number of the restored epoch.
        int: The number of the restored iteration.

    Raises:
        ModelRestoreError: An error occurred loading model.
    """

    def find_fname(model_path):
        f_list = sorted(os.listdir(model_path))
        ep = 0
        it = 0
        for f in f_list:
            if f.endswith(".npy"):
                nums = f[:-4].split("_")
                ep = max(ep, int(nums[0]))
                it = max(it, int(nums[1]))
        fname = str(ep) + "_" + str(it) + ".npy"
        return fname

    try:
        fname = (
            "model.npy"
            if os.path.exists("{0}/model.npy".format(model_path))
            else find_fname(model_path)
        )
        checkpoint = np.load("{0}/{1}".format(model_path, fname), allow_pickle=True)
    except:
        raise ModelRestoreError("Cannot find the model.")
    try:
        circuit_dict = checkpoint.item()["circuit"]
        pargs = circuit_dict["pargs"]
        grads = circuit_dict["grads"]
        identity = circuit_dict["identity"]
        params = Variable(pargs, grads, identity=identity)
        assert params.shape == model.params.shape, ModelRestoreError(
            "Model does not match the network."
        )
        model.params = params
        model.update()

        if restore_optim:
            optim_dict = checkpoint.item()["optimizer"]
            optim_lodeder = OptimizerInitializer(optim_dict)
            optim = optim_lodeder()
            model.optimizer = optim
    except:
        raise ModelRestoreError("Cannot load the model correctly.")

    ep = checkpoint.item()["epoch"]
    it = checkpoint.item()["iter"]

    logger.info(f"Successfully restored checkpoint at ep: {ep} it: {it}")
    return ep, it + 1
