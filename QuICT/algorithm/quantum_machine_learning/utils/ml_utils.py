import os
import torch
import shutil
import numpy as np
import random


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
        it (int): The number of the saved iteration.
        latest (bool, optional): Whether this is the last iteration. Defaults to False.
    """
    os.makedirs(model_path, exist_ok=True)
    checkpoint = dict(
        epoch=ep,
        iter=it,
        graph=net.state_dict(),
        optimizer=optim.state_dict(),
    )
    torch.save(checkpoint, "{0}/model.ckpt".format(model_path))
    if not latest:
        shutil.copy(
            "{0}/model.ckpt".format(model_path),
            "{0}/{1}_{2}.ckpt".format(model_path, ep, it),
        )


def restore_checkpoint(net, optim, model_path, device, resume):
    """Restore the model and optimizer from a checkpoint.

    Args:
        resume (int/bool): If resume is True, restore the latest checkpoint.
            Or users can specify a checkpoint saved in an iteration to restore.

    Returns:
        int: The number of the restored iteration.
    """
    assert resume and model_path
    try:
        model_name = (
            "{0}/model.ckpt".format(model_path)
            if resume is True
            else "{0}/{1}_{2}.ckpt".format(model_path, resume["ep"], resume["it"])
        )
    except:
        raise Exception("Cannot find the model.")

    checkpoint = torch.load(model_name, map_location=device)
    try:
        net.load_state_dict(checkpoint["graph"])
        if optim:
            optim.load_state_dict(checkpoint["optimizer"])
    except:
        raise Exception("Cannot load the model correctly.")

    ep = checkpoint["ep"]
    it = checkpoint["iter"]
    assert resume is True or (resume["ep"] == ep and resume["it"] == it)

    return ep, it


def set_optimizer(optimizer, net, lr):
    assert optimizer in OPTIMIZER_LIST
    optimizer = getattr(torch.optim, optimizer)
    optim = optimizer([dict(params=net.parameters(), lr=lr)])
    return optim
