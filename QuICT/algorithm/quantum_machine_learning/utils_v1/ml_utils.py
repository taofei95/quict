import os
import time
import torch
import shutil
import numpy as np
import random

from QuICT.core.gate.utils import Variable
from QuICT.tools.logger import *
from QuICT.algorithm.quantum_machine_learning.tools.exception import *

logger = Logger("ML_utils")


def set_seed(seed: int):
    """Set random seed.

    Args:
        seed (int): The random seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_checkpoint(net, optim, model_path, ep, it, latest=False):
    """Save the model and optimizer as a checkpoint.

    Args:
        net (torch.nn.Module): The network that to be saved.
        optim (torch.optim): The optimizer that to be saved.
        model_path (str): The path to the saved checkpoint.
        ep (int): The number of the saved epoch.
        it (int): The number of the saved iteration.
        latest (bool, optional): Whether this is the last iteration. Defaults to False.
    """
    os.makedirs(model_path, exist_ok=True)
    checkpoint = dict(
        epoch=ep, iter=it, graph=net.state_dict(), optimizer=optim.state_dict(),
    )
    torch.save(checkpoint, "{0}/model.ckpt".format(model_path))
    if not latest:
        shutil.copy(
            "{0}/model.ckpt".format(model_path),
            "{0}/{1}_{2}.ckpt".format(model_path, ep, it),
        )


def restore_checkpoint(net, optim, model_path, device):
    """Restore the model and optimizer from a checkpoint.

    Args:
        net (torch.nn.Module): The network that to be restored.
        optim (torch.optim): The optimizer that to be restored.
        model_path (str): The path to the saved checkpoint.
        device (torch.device): The device to which the model is assigned.

    Returns:
        int: The number of the restored epoch.
        int: The number of the restored iteration.
    """
    try:
        checkpoint = torch.load(model_path, map_location=device)
    except:
        raise ModelRestoreError("Cannot find the model.")
    try:
        net.load_state_dict(checkpoint["graph"])
        if optim:
            optim.load_state_dict(checkpoint["optimizer"])
    except:
        raise ModelRestoreError("Cannot load the model correctly.")

    ep = checkpoint["epoch"]
    it = checkpoint["iter"]

    logger.info(f"Successfully restored checkpoint at ep: {ep} it: {it}")
    return ep, it


# def apply_optimizer(optimizer: tf.keras.optimizers.Optimizer, variables: Variable):
#     tfvariable_list = convert_to_tfvariable(variables.pargs)
#     optimizer.apply_gradients(zip(variables.grads, tfvariable_list), experimental_aggregate_gradients=True)
#     pargs = convert_to_numpy(tfvariable_list)
#     variables.pargs = pargs
#     return variables


# def convert_to_tfvariable(pargs: np.ndarray):
#     pargs_list = []
#     for parg in pargs:
#         pargs_list.append(tf.Variable(parg))
#     return pargs_list


# def convert_to_numpy(variable_list: list):
#     pargs_list = []
#     for variable in variable_list:
#         pargs_list.append(variable.numpy())
#     return np.array(pargs_list)
