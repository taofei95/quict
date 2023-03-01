import collections
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import tqdm
import time
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.append("/home/zoker/quict")

from QuICT.algorithm.quantum_machine_learning.ansatz_library import QNNLayer
from QuICT.algorithm.quantum_machine_learning.utils.encoding import *
from QuICT.algorithm.quantum_machine_learning.utils.ml_utils import *
from QuICT.algorithm.quantum_machine_learning.model.QNN import QuantumNet
from QuICT.simulation.state_vector import ConstantStateVectorSimulator


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
    X = torch.from_numpy(np.array(X_rmcon))
    Y = torch.from_numpy(np.array(Y_rmcon))
    return X, Y


nocon_train_X, nocon_train_Y = remove_conflict(resized_train_X, train_Y, (4, 4))
nocon_test_X, nocon_test_Y = remove_conflict(resized_test_X, test_Y, (4, 4))
print("Remaining training examples: ", len(nocon_train_Y))
print("Remaining testing examples: ", len(nocon_test_Y))


def binary_img(X, threshold):
    X = X > threshold
    X = X.type(torch.int)
    return X


threshold = 0.5
bin_train_X = binary_img(nocon_train_X, threshold)
bin_test_X = binary_img(nocon_test_X, threshold)
device = torch.device("cuda:0")

train_X = bin_train_X.to(device)
train_Y = nocon_train_Y.to(device)
test_X = bin_test_X.to(device)
test_Y = nocon_test_Y.to(device)

# pqc = QNNLayer(list(range(4)), 4, device=device)
# params = nn.Parameter(torch.rand(1, 4, device=device), requires_grad=True)
# model_circuit = pqc.circuit_layer(["XX"], params)
# model_circuit.draw()

EPOCH = 3  # 训练总轮数
BATCH_SIZE = 32  # 一次迭代使用的样本数
LR = 0.001  # 梯度下降的学习率
SEED = 0  # 随机数种子

set_seed(SEED)  # 设置全局随机种子

train_dataset = data.TensorDataset(train_X, train_Y)
test_dataset = data.TensorDataset(test_X, test_Y)
train_loader = data.DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
)
test_loader = data.DataLoader(
    dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
)

layers = ["XX", "ZZ", "XX", "ZZ", "XX", "ZZ", "XX", "ZZ"]
net = QuantumNet(4, layers, encoding="FRQI", device=device)
optim = torch.optim.Adam([dict(params=net.parameters(), lr=LR)])


def loss_func(y_true, y_pred):
    y_true = 2 * y_true.type(torch.float32) - 1.0
    y_pred = 2 * y_pred - 1.0
    loss = torch.clamp(1 - y_pred * y_true, min=0.0)
    correct = torch.where(y_true * y_pred > 0)[0].shape[0]
    return torch.mean(loss), correct


import torch.utils.tensorboard

now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
dist_dir = "QNN_MNIST_FRQI_" + now

tb = torch.utils.tensorboard.SummaryWriter(log_dir=dist_dir + "/logs")

# train epoch
for ep in range(EPOCH):
    net.train()
    loader = tqdm.tqdm(
        train_loader, desc="Training epoch {}".format(ep + 1), leave=True
    )
    # train iteration
    for it, (x_train, y_train) in enumerate(loader):
        optim.zero_grad()
        y_pred = net(x_train)
        loss, correct = loss_func(y_train, y_pred)
        accuracy = correct / len(y_train)
        loss.backward()
        optim.step()
        loader.set_postfix(
            it=it,
            loss="{:.3f}".format(loss),
            accuracy="{:.3f}".format(accuracy),
        )
        tb.add_scalar("train/loss", loss, ep * len(loader) + it)
        tb.add_scalar("train/accuracy", accuracy, ep * len(loader) + it)

        # Save checkpoint
        latest = (ep + 1) == EPOCH and (it + 1) == len(loader)
        if (it != 0 and it % 30 == 0) or latest:
            save_checkpoint(net, optim, dist_dir, ep, it, latest)

    # Validation
    net.eval()
    loader_val = tqdm.tqdm(
        test_loader, desc="Validating epoch {}".format(ep + 1), leave=True
    )
    loss_val_list = []
    total_correct = 0
    for it, (x_test, y_test) in enumerate(loader_val):
        y_pred = net(x_test)
        loss_val, correct = loss_func(y_test, y_pred)
        loss_val_list.append(loss_val.cpu().detach().numpy())
        total_correct += correct
        accuracy_val = correct / len(y_test)
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
