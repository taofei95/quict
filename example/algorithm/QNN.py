import collections
import torch
import torch.utils.data as data
import tqdm
from torchvision import datasets, transforms

from QuICT.algorithm.quantum_machine_learning.utils.encoding import *
from QuICT.algorithm.quantum_machine_learning.utils.ml_utils import *
from QuICT.algorithm.quantum_machine_learning.model.QNN import QuantumNet


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
    X = torch.from_numpy(np.array(X_rmcon))
    Y = torch.from_numpy(np.array(Y_rmcon))
    return X, Y


def binary_img(X, threshold):
    X = (X > threshold).type(torch.int)
    return X


def loss_func(y_true, y_pred):
    y_true = 2 * y_true.type(torch.float32) - 1.0
    y_pred = 2 * y_pred - 1.0
    loss = torch.clamp(1 - y_pred * y_true, min=0.0)
    correct = torch.where(y_true * y_pred > 0)[0].shape[0]
    return torch.mean(loss), correct


# data loading and preprocessing
BATCH_SIZE = 32
THRESHOLD = 0.5
device = torch.device("cuda:0")

# load data
train_data = datasets.MNIST(root="./data/", train=True, download=True)
test_data = datasets.MNIST(root="./data/", train=False, download=True)
train_X = train_data.data
train_Y = train_data.targets
test_X = test_data.data
test_Y = test_data.targets
# filter two classes
train_X, train_Y = filter_targets(train_X, train_Y)
test_X, test_Y = filter_targets(test_X, test_Y)
# resize
resized_train_X = downscale(train_X, (4, 4))
resized_test_X = downscale(test_X, (4, 4))
# remove conflicts
nocon_train_X, nocon_train_Y = remove_conflict(resized_train_X, train_Y, (4, 4))
nocon_test_X, nocon_test_Y = remove_conflict(resized_test_X, test_Y, (4, 4))
# binarization
bin_train_X = binary_img(nocon_train_X, THRESHOLD)
bin_test_X = binary_img(nocon_test_X, THRESHOLD)
# generate dataloader
train_X = bin_train_X.to(device)
train_Y = nocon_train_Y.to(device)
test_X = bin_test_X.to(device)
test_Y = nocon_test_Y.to(device)
train_dataset = data.TensorDataset(train_X, train_Y)
test_dataset = data.TensorDataset(test_X, test_Y)
train_loader = data.DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
)
test_loader = data.DataLoader(
    dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
)

EPOCH = 3
LR = 0.001
SEED = 17
set_seed(SEED)

net = QuantumNet(16, ["XX", "ZZ"], encoding="qubit", device=device)
optim = torch.optim.Adam([dict(params=net.parameters(), lr=LR)])

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
    print("Validation Average Loss: {}, Accuracy: {}".format(avg_loss, avg_acc))
