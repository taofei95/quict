import collections
import tqdm
from torchvision import datasets, transforms
import sys
import numpy_ml
import yaml
import time

from QuICT.algorithm.quantum_machine_learning.encoding import *
from QuICT.algorithm.quantum_machine_learning.utils.loss import *
from QuICT.algorithm.quantum_machine_learning.utils.ml_utils import *
from QuICT.algorithm.quantum_machine_learning.model.QNN import QuantumNet
from QuICT.algorithm.quantum_machine_learning.utils.data import *
from QuICT.algorithm.quantum_machine_learning.ansatz_library import *


def filter_targets(X, Y, class0=3, class1=6):
    idx = (Y == class0) | (Y == class1)
    X, Y = (X[idx], Y[idx])
    Y = Y == class1
    return X, Y


def downscale(X, resize):
    transform = transforms.Resize(size=resize)
    X = transform(X) / 255.0
    return X


def remove_conflict(X, Y, resize):
    x_dict = collections.defaultdict(set)
    for x, y in zip(X, Y):
        x_dict[tuple(x.numpy().flatten())].add(y.item())
    X_rmcon = []
    Y_rmcon = []
    for x in x_dict.keys():
        if len(x_dict[x]) == 1:
            X_rmcon.append(np.array(x).reshape(resize))
            Y_rmcon.append(list(x_dict[x])[0])
    X = np.array(X_rmcon)
    Y = np.array(Y_rmcon)
    return X, Y


def binary_img(X, threshold):
    X = X > threshold
    X = X.astype(np.int16)
    return X


def encoding_img(X, encoding):
    data_circuits = []
    for i in tqdm.tqdm(range(len(X))):
        data_circuit = encoding(X[i])
        data_circuits.append(data_circuit)
    return data_circuits


def train_iteration(it, net, tb, loader, x_train, y_train, optimizer, loss_fun):
    loss, correct = net.run_step(x_train, y_train)
    accuracy = correct / len(y_train)
    loader.set_postfix(
        it=it, loss="{:.3f}".format(loss), accuracy="{:.3f}".format(accuracy)
    )
    tb.add_scalar("train/loss", loss, ep * len(loader) + it)
    tb.add_scalar("train/accuracy", accuracy, ep * len(loader) + it)
    # Save checkpoint
    latest = (ep + 1) == EPOCH and (it + 1) == len(loader)
    if (it != 0 and it % 30 == 0) or latest:
        save_checkpoint(net, optimizer, model_path, ep, it, latest)


def train(ep, it_start, net, tb):
    loader = tqdm.tqdm(
        train_loader, desc="Training epoch {}".format(ep + 1), leave=True
    )
    for it, (x_train, y_train) in enumerate(loader):
        if it < it_start:
            continue
        it_start = 0
        train_iteration(it, net, tb, loader, x_train, y_train, optimizer, loss_fun)


def validate(ep, net, tb):
    loader_val = tqdm.tqdm(
        test_loader, desc="Validating epoch {}".format(ep + 1), leave=True
    )
    loss_val_list = []
    total_correct = 0
    for it, (x_test, y_test) in enumerate(loader_val):
        loss_val, correct_val = net.run_step(x_test, y_test, train=False)
        loss_val_list.append(loss_val)
        total_correct += correct_val
        accuracy_val = correct_val / len(y_test)
        loader_val.set_postfix(
            it=it,
            loss="{:.3f}".format(loss_val),
            accuracy="{:.3f}".format(accuracy_val),
        )

    avg_loss = np.mean(loss_val_list)
    avg_acc = total_correct / (len(loader_val) * BATCH_SIZE)
    tb.add_scalar("validation/loss", avg_loss, ep)
    tb.add_scalar("validation/accuracy", avg_acc, ep)
    print("Validation Average Loss: {}, Accuracy: {}".format(avg_loss, avg_acc))


if __name__ == "__main__":
    RESIZE = (8, 8)

    train_data = datasets.MNIST(root="./data/", train=True, download=True)
    test_data = datasets.MNIST(root="./data/", train=False, download=True)
    train_X = train_data.data
    train_Y = train_data.targets
    test_X = test_data.data
    test_Y = test_data.targets
    print("Training examples: ", len(train_Y))
    print("Testing examples: ", len(test_Y))

    train_X, train_Y = filter_targets(train_X, train_Y)
    test_X, test_Y = filter_targets(test_X, test_Y)
    print("Filtered training examples: ", len(train_Y))
    print("Filtered testing examples: ", len(test_Y))

    resized_train_X = downscale(train_X, RESIZE)
    resized_test_X = downscale(test_X, RESIZE)

    nocon_train_X, nocon_train_Y = remove_conflict(resized_train_X, train_Y, RESIZE)
    nocon_test_X, nocon_test_Y = remove_conflict(resized_test_X, test_Y, RESIZE)
    print("Remaining training examples: ", len(nocon_train_Y))
    print("Remaining testing examples: ", len(nocon_test_Y))

    threshold = 0.5
    bin_train_X = binary_img(nocon_train_X, threshold)
    bin_test_X = binary_img(nocon_test_X, threshold)

    EPOCH = 10
    BATCH_SIZE = 32
    LR = 0.001
    SEED = 17
    ep_start = 0
    it_start = 0
    GRAYSCALE = 2

    set_seed(SEED)

    encoding = FRQI(GRAYSCALE)
    train_X = encoding_img(bin_train_X, encoding)
    test_X = encoding_img(bin_test_X, encoding)
    train_Y = nocon_train_Y
    test_Y = nocon_test_Y

    train_dataset = Dataset(train_X, train_Y)
    test_dataset = Dataset(test_X, test_Y)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )

    loss_fun = HingeLoss()
    optimizer = numpy_ml.neural_nets.optimizers.Adam(lr=LR)
    n_qubits = int(np.log2(RESIZE[0] * RESIZE[1])) + 2
    ansatz = CRADL(n_qubits, 6)
    net = QuantumNet(
        n_qubits=n_qubits,
        ansatz=ansatz,
        optimizer=optimizer,
        loss_fun=loss_fun,
        device="CPU",
    )

    import torch.utils.tensorboard

    now_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    model_path = "./QNN2.0_MNIST_" + now_time + "/"
    tb = torch.utils.tensorboard.SummaryWriter(log_dir=model_path + "logs")

    # save settings
    config = {
        "encoding": encoding,
        "model circuit": ansatz,
        "grayscale": GRAYSCALE,
        "resize": RESIZE,
        "LR": LR,
        "loss_fun": loss_fun,
    }
    config_file = open(model_path + "config.yaml", "w")
    config_file.write(yaml.dump(config))
    config_file.close()

    # ep_start, it_start, optimizer = restore_checkpoint(net, model_path, restore_optim=True)

    for ep in range(ep_start, EPOCH):
        train(ep, it_start, net, tb)
        validate(ep, net, tb)
