import collections
import tqdm
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import sys
import numpy_ml
import time

sys.path.append("/home/zoker/quict")

from QuICT.algorithm.quantum_machine_learning.encoding import *
from QuICT.algorithm.quantum_machine_learning.utils.loss import *
from QuICT.algorithm.quantum_machine_learning.utils.ml_utils import *
from QuICT.algorithm.quantum_machine_learning.model.QNN import QuantumNet
from QuICT.algorithm.quantum_machine_learning.data import *

train_data = datasets.MNIST(root="./data/", train=True, download=True)
test_data = datasets.MNIST(root="./data/", train=False, download=True)
train_X = train_data.data
train_Y = train_data.targets
test_X = test_data.data
test_Y = test_data.targets
print("Training examples: ", len(train_Y))
print("Testing examples: ", len(test_Y))


def filter_targets(X, Y, class0=3, class1=6):
    idx = (Y == class0) | (Y == class1)
    X, Y = (X[idx], Y[idx])
    Y = Y == class1
    return X, Y


train_X, train_Y = filter_targets(train_X, train_Y)
test_X, test_Y = filter_targets(test_X, test_Y)
print("Filtered training examples: ", len(train_Y))
print("Filtered testing examples: ", len(test_Y))
print("Label: ", train_Y[200])
plt.imshow(train_X[200], cmap="gray")


def downscale(X, resize):
    transform = transforms.Resize(size=resize)
    X = transform(X) / 255.0
    return X


resized_train_X = downscale(train_X, (4, 4))
resized_test_X = downscale(test_X, (4, 4))
plt.imshow(resized_train_X[200], cmap="gray")


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


nocon_train_X, nocon_train_Y = remove_conflict(resized_train_X, train_Y, (4, 4))
nocon_test_X, nocon_test_Y = remove_conflict(resized_test_X, test_Y, (4, 4))
print("Remaining training examples: ", len(nocon_train_Y))
print("Remaining testing examples: ", len(nocon_test_Y))


def binary_img(X, threshold):
    X = X > threshold
    X = X.astype(np.int16)
    return X


threshold = 0.5
bin_train_X = binary_img(nocon_train_X, threshold)
bin_test_X = binary_img(nocon_test_X, threshold)


def encoding_img(X, encoding):
    data_circuits = []
    for x in X:
        data_circuit = encoding(x)
        data_circuits.append(data_circuit)

    return data_circuits


EPOCH = 3  # 训练总轮数
BATCH_SIZE = 32  # 一次迭代使用的样本数
LR = 0.001  # 梯度下降的学习率
SEED = 17  # 随机数种子

set_seed(SEED)

# encoding = Qubit(16)
# train_X = encoding_img(bin_train_X, encoding)
# test_X = encoding_img(bin_test_X, encoding)
train_X = bin_train_X
test_X = bin_test_X
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
# net = QuantumNet(n_qubits=17, readout=16)
net = QuantumNet(n_qubits=6, readout=5)

import torch.utils.tensorboard

now_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
model_path = "/home/zoker/quict/QNN2.0_MNIST_" + now_time + "/"
tb = torch.utils.tensorboard.SummaryWriter(log_dir=model_path + "logs")

encoding = FRQI(2)
# encoding = Qubit(16)

# train epoch
for ep in range(EPOCH):
    loader = tqdm.tqdm(
        train_loader, desc="Training epoch {}".format(ep + 1), leave=True
    )
    # train iteration
    for it, (x_train, y_train) in enumerate(loader):
        x_train = [encoding(x) for x in x_train]
        loss, correct = net.run_step(x_train, y_train, optimizer, loss_fun)
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

    # Validation
    loader_val = tqdm.tqdm(
        test_loader, desc="Validating epoch {}".format(ep + 1), leave=True
    )
    loss_val_list = []
    total_correct = 0
    for it, (x_test, y_test) in enumerate(loader_val):
        loss_val, correct_val = net.run_step(
            x_train, y_train, optimizer, loss_fun, train=False
        )
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

"""
Training epoch 1: 100%|██████████████████████████████████████████████████████████████████| 323/323 [07:14<00:00,  1.35s/it, accuracy=0.688, it=322, loss=0.792]
Validating epoch 1: 100%|███████████████████████████████████████████████████████████████████| 56/56 [00:51<00:00,  1.09it/s, accuracy=0.688, it=55, loss=0.790]
Validation Average Loss: 0.7902488731873062, Accuracy: 0.6875
Training epoch 2: 100%|██████████████████████████████████████████████████████████████████| 323/323 [14:11<00:00,  2.64s/it, accuracy=0.875, it=322, loss=0.470]
Validating epoch 2: 100%|███████████████████████████████████████████████████████████████████| 56/56 [01:37<00:00,  1.74s/it, accuracy=0.875, it=55, loss=0.469]
Validation Average Loss: 0.46901776041853127, Accuracy: 0.875
Training epoch 3: 100%|██████████████████████████████████████████████████████████████████| 323/323 [20:37<00:00,  3.83s/it, accuracy=0.938, it=322, loss=0.326]
Validating epoch 3: 100%|███████████████████████████████████████████████████████████████████| 56/56 [02:20<00:00,  2.51s/it, accuracy=0.938, it=55, loss=0.326]
Validation Average Loss: 0.32613962477589203, Accuracy: 0.9375
"""
