import collections
import tqdm
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import sys
import numpy_ml
import yaml
import time

sys.path.append("/home/zoker/quict")

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


def train(ep, it_start, net, tb):
    loader = tqdm.tqdm(
        train_loader, desc="Training epoch {}".format(ep + 1), leave=True
    )
    for it, (x_train, y_train) in enumerate(loader):
        if it < it_start:
            continue
        it_start = 0
        expectations = net.forward(x_train)  # (32, 1)
        y_true = (2 * y_train - 1.0).reshape(expectations.shape)
        y_pred = -expectations
        loss = loss_fun(y_pred, y_true)
        # optimize
        net.backward(loss)
        # update
        net.update()

        correct = np.where(y_true * y_pred.pargs > 0)[0].shape[0]
        accuracy = correct / len(y_train)
        loader.set_postfix(
            it=it, loss="{:.3f}".format(loss.item), accuracy="{:.3f}".format(accuracy)
        )
        tb.add_scalar("train/loss", loss.item, ep * len(loader) + it)
        tb.add_scalar("train/accuracy", accuracy, ep * len(loader) + it)
        # Save checkpoint
        latest = (ep + 1) == EPOCH and (it + 1) == len(loader)
        if (it != 0 and it % 30 == 0) or latest:
            save_checkpoint(net, optimizer, model_path, ep, it, latest)


def validate(ep, net, tb):
    loader_val = tqdm.tqdm(
        test_loader, desc="Validating epoch {}".format(ep + 1), leave=True
    )
    loss_val_list = []
    total_correct = 0
    for it, (x_test, y_test) in enumerate(loader_val):
        expectations_val = net.forward(x_test, train=False)
        y_true_val = (2 * y_test - 1.0).reshape(expectations_val.shape)
        y_pred_val = -expectations_val
        loss_val = loss_fun(y_pred_val, y_true_val)

        loss_val_list.append(loss_val.item)
        correct_val = np.where(y_true_val * y_pred_val.pargs > 0)[0].shape[0]
        total_correct += correct_val
        accuracy_val = correct_val / len(y_test)
        loader_val.set_postfix(
            it=it,
            loss="{:.3f}".format(loss_val.item),
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

    EPOCH = 5  # 训练总轮数
    BATCH_SIZE = 32  # 一次迭代使用的样本数
    LR = 0.001  # 梯度下降的学习率
    SEED = 17  # 随机数种子
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
    ansatz = HEAnsatz(n_qubits - 1, 5, ["RZ", "RY", "RZ", "CRy"], readout=[0])
    # ansatz = CRADL(n_qubits, 6)
    net = QuantumNet(
        n_qubits=n_qubits - 1,
        ansatz=ansatz,
        optimizer=optimizer,
        device="CPU",
        data_qubits=list(range(n_qubits - 1))
    )

    import torch.utils.tensorboard

    now_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    model_path = "/home/zoker/quict/QNN2.0_MNIST_" + now_time + "/"
    # model_path = "/home/zoker/quict/QNN2.0_MNIST_2023-07-27-16_52_50/"
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
        if ep > ep_start:
            it_start = 0
        train(ep, it_start, net, tb)
        validate(ep, net, tb)


"""QNN2.0_MNIST_2023-07-11-17_33_55
Training epoch 1: 100%|██████████████████████████████████████████████████████████████████████████████████████| 376/376 [07:28<00:00,  1.19s/it, accuracy=0.781, it=375, loss=0.930]
Validating epoch 1: 100%|███████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:20<00:00,  2.96it/s, accuracy=0.812, it=60, loss=0.907]
Validation Average Loss: 0.9229925045286855, Accuracy: 0.8370901639344263
Training epoch 2: 100%|██████████████████████████████████████████████████████████████████████████████████████| 376/376 [07:25<00:00,  1.19s/it, accuracy=0.875, it=375, loss=0.915]
Validating epoch 2: 100%|███████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:20<00:00,  2.92it/s, accuracy=0.938, it=60, loss=0.910]
Validation Average Loss: 0.9035673828734119, Accuracy: 0.9113729508196722
Training epoch 3: 100%|██████████████████████████████████████████████████████████████████████████████████████| 376/376 [07:24<00:00,  1.18s/it, accuracy=0.906, it=375, loss=0.906]
Validating epoch 3: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:20<00:00,  2.93it/s, accuracy=0.938, it=60, loss=0.914]
Validation Average Loss: 0.8919897259585313, Accuracy: 0.8673155737704918
Training epoch 4: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 376/376 [07:21<00:00,  1.17s/it, accuracy=0.906, it=375, loss=0.900]
Validating epoch 4: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:21<00:00,  2.87it/s, accuracy=0.938, it=60, loss=0.904]
Validation Average Loss: 0.8853000620562105, Accuracy: 0.8872950819672131
Training epoch 5: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 376/376 [07:22<00:00,  1.18s/it, accuracy=0.906, it=375, loss=0.895]
Validating epoch 5: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:21<00:00,  2.90it/s, accuracy=0.906, it=60, loss=0.899]
Validation Average Loss: 0.8810967802415437, Accuracy: 0.8995901639344263
Training epoch 6: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 376/376 [07:21<00:00,  1.18s/it, accuracy=0.906, it=375, loss=0.891]
Validating epoch 6: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:20<00:00,  2.93it/s, accuracy=0.906, it=60, loss=0.897]
Validation Average Loss: 0.8777057849065354, Accuracy: 0.8975409836065574
Training epoch 7: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 376/376 [07:21<00:00,  1.17s/it, accuracy=0.938, it=375, loss=0.888]
Validating epoch 7: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:20<00:00,  3.00it/s, accuracy=0.906, it=60, loss=0.896]
Validation Average Loss: 0.8748023455117792, Accuracy: 0.8954918032786885
Training epoch 8: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 376/376 [07:23<00:00,  1.18s/it, accuracy=0.938, it=375, loss=0.887]
Validating epoch 8: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:20<00:00,  3.00it/s, accuracy=0.906, it=60, loss=0.893]
Validation Average Loss: 0.872750128822618, Accuracy: 0.8939549180327869
Training epoch 9: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 376/376 [07:23<00:00,  1.18s/it, accuracy=0.938, it=375, loss=0.885]
Validating epoch 9: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:20<00:00,  2.94it/s, accuracy=0.906, it=60, loss=0.890]
Validation Average Loss: 0.8708665872102664, Accuracy: 0.8975409836065574
Training epoch 10: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 376/376 [07:23<00:00,  1.18s/it, accuracy=0.938, it=375, loss=0.884]
Validating epoch 10: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:20<00:00,  2.95it/s, accuracy=0.906, it=60, loss=0.888]
Validation Average Loss: 0.8689571453620687, Accuracy: 0.9006147540983607
"""


"""
Training epoch 1: 100%|██████████████████████████████████████████████████████████████████████████████████████| 376/376 [16:36<00:00,  2.65s/it, accuracy=0.562, it=375, loss=0.969]
Validating epoch 1: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:52<00:00,  1.17it/s, accuracy=0.438, it=60, loss=0.977]
Validation Average Loss: 0.9586349541978338, Accuracy: 0.6659836065573771
Training epoch 2: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 376/376 [16:33<00:00,  2.64s/it, accuracy=0.906, it=375, loss=0.946]
Validating epoch 2: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:52<00:00,  1.16it/s, accuracy=1.000, it=60, loss=0.941]
Validation Average Loss: 0.9311716799123593, Accuracy: 0.9590163934426229
Training epoch 3: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 376/376 [16:30<00:00,  2.64s/it, accuracy=0.906, it=375, loss=0.939]
Validating epoch 3: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:52<00:00,  1.17it/s, accuracy=1.000, it=60, loss=0.934]
Validation Average Loss: 0.9234953700769899, Accuracy: 0.9579918032786885
Training epoch 4: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 376/376 [16:28<00:00,  2.63s/it, accuracy=0.875, it=375, loss=0.935]
Validating epoch 4: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:51<00:00,  1.19it/s, accuracy=1.000, it=60, loss=0.935]
Validation Average Loss: 0.9187034125666818, Accuracy: 0.9364754098360656
Training epoch 5: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 376/376 [16:15<00:00,  2.60s/it, accuracy=0.875, it=375, loss=0.931]
Validating epoch 5: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:51<00:00,  1.18it/s, accuracy=0.875, it=60, loss=0.938]
Validation Average Loss: 0.9156545732699695, Accuracy: 0.8862704918032787
Training epoch 6: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 376/376 [16:15<00:00,  2.59s/it, accuracy=0.844, it=375, loss=0.928]
Validating epoch 6: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:50<00:00,  1.22it/s, accuracy=0.844, it=60, loss=0.938]
Validation Average Loss: 0.9134938899093542, Accuracy: 0.875
Training epoch 7: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 376/376 [16:13<00:00,  2.59s/it, accuracy=0.844, it=375, loss=0.926]
Validating epoch 7: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:49<00:00,  1.22it/s, accuracy=0.844, it=60, loss=0.937]
Validation Average Loss: 0.9120725299141247, Accuracy: 0.8867827868852459
Training epoch 8: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 376/376 [16:15<00:00,  2.59s/it, accuracy=0.875, it=375, loss=0.924]
Validating epoch 8: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:51<00:00,  1.20it/s, accuracy=0.844, it=60, loss=0.934]
Validation Average Loss: 0.9107190977007117, Accuracy: 0.9052254098360656
Training epoch 9: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 376/376 [16:10<00:00,  2.58s/it, accuracy=0.875, it=375, loss=0.921]
Validating epoch 9: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:51<00:00,  1.19it/s, accuracy=0.969, it=60, loss=0.929]
Validation Average Loss: 0.9089419571439197, Accuracy: 0.9349385245901639
Training epoch 10: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 376/376 [16:11<00:00,  2.58s/it, accuracy=0.938, it=375, loss=0.918]
Validating epoch 10: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:52<00:00,  1.17it/s, accuracy=1.000, it=60, loss=0.923]
Validation Average Loss: 0.9060153970942376, Accuracy: 0.9564549180327869
"""
