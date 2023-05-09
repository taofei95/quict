import os
import shutil
import numpy as np
import random
import numpy_ml

from QuICT.core.utils import Variable
from QuICT.tools.logger import *
from QuICT.algorithm.quantum_machine_learning.tools.exception import *

logger = Logger("ML_utils")


def set_seed(seed: int):
    """Set random seed.

    Args:
        seed (int): The random seed.
    """
    np.random.seed(seed)
    random.seed(seed)


def save_checkpoint(model, optim, model_path, ep, it, latest=False):
    os.makedirs(model_path, exist_ok=True)
    circuit_dict = dict(
        pargs=model.params.pargs,
        grads=model.params.grads,
        identity=model.params.identity,
        shape=model.params.shape,
    )
    optim_dict = dict(cache=optim.cache, hyperparameters=optim.hyperparameters)
    checkpoint = dict(epoch=ep, iter=it, circuit=circuit_dict, optimizer=optim_dict)
    np.save("{0}/model.npy".format(model_path), checkpoint)

    if not latest:
        shutil.copy(
            "{0}/model.npy".format(model_path),
            "{0}/{1}_{2}.npy".format(model_path, ep, it),
        )


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


def restore_checkpoint(model, model_path, restore_optim=True):
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
        assert params.shape == model.params.shape
        model.params = params
        model.update()
        optim = None

        if restore_optim:
            optim_dict = checkpoint.item()["optimizer"]
            optim_lodeder = numpy_ml.neural_nets.initializers.OptimizerInitializer(
                optim_dict
            )
            optim = optim_lodeder()
    except:
        raise ModelRestoreError("Cannot load the model correctly.")

    ep = checkpoint.item()["epoch"]
    it = checkpoint.item()["iter"]

    logger.info(f"Successfully restored checkpoint at ep: {ep} it: {it}")
    return ep, it, optim


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

if __name__ == "__main__":
    # optim = numpy_ml.neural_nets.optimizers.Adam(lr=0.1)
    # optim_dict = dict(cache=optim.cache, hyperparameters=optim.hyperparameters)

    # optim_lodeder = numpy_ml.neural_nets.initializers.OptimizerInitializer(optim_dict)
    # print(optim_lodeder.param)
    # optim_loded = optim_lodeder()
    # print(optim_loded)

    from QuICT.algorithm.quantum_machine_learning.utils import Hamiltonian
    from QuICT.algorithm.quantum_machine_learning.utils.ml_utils import *
    from QuICT.algorithm.tools.drawer.graph_drawer import *
    from QuICT.algorithm.quantum_machine_learning.model import QAOA

    n = 5
    p = 4
    nodes = list(range(n))
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 3], [2, 4]]

    def maxcut_hamiltonian(edges):
        pauli_list = []
        for edge in edges:
            pauli_list.append([-1.0, "Z" + str(edge[0]), "Z" + str(edge[1])])
        hamiltonian = Hamiltonian(pauli_list)

        return hamiltonian

    H = maxcut_hamiltonian(edges)

    qaoa_net = QAOA(n_qubits=n, p=p, hamiltonian=H)
    # optim = numpy_ml.neural_nets.optimizers.Adam(lr=0.1)
    ep, it, optim = restore_checkpoint(qaoa_net, "/home/zoker/quict/test_save/", restore_optim=False)
