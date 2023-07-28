import collections
import tqdm
from torchvision import datasets, transforms

from QuICT.algorithm.quantum_machine_learning.encoding import *
from QuICT.algorithm.quantum_machine_learning.utils.loss import *
from QuICT.algorithm.quantum_machine_learning.utils.ml_utils import *
from QuICT.algorithm.quantum_machine_learning.utils.data import *


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


def change_grayscale(X, grayscale):
    assert grayscale >= 2 and grayscale <= 256
    grays = np.linspace(0, 1, grayscale)
    thresholds = np.linspace(0, 1, grayscale + 1)
    for i in range(grayscale):
        X[np.logical_and(X >= thresholds[i], X <= thresholds[i + 1])] = grays[i]
    return X


def encoding_img(X, encoding):
    data_circuits = []
    for i in tqdm.tqdm(range(len(X))):
        data_circuit = encoding(X[i])
        data_circuits.append(data_circuit)
    return data_circuits


def get_data_loader(resize, encoding=None, batch_size=32, binary=True, max_size=20):
    train_data = datasets.MNIST(
        root="/home/zoker/quict/data/", train=True, download=True
    )
    test_data = datasets.MNIST(
        root="/home/zoker/quict/data/", train=False, download=True
    )
    train_X = train_data.data
    train_Y = train_data.targets
    test_X = test_data.data
    test_Y = test_data.targets

    train_X, train_Y = filter_targets(train_X, train_Y)
    test_X, test_Y = filter_targets(test_X, test_Y)

    train_X = downscale(train_X, resize)
    test_X = downscale(test_X, resize)

    train_X, train_Y = remove_conflict(train_X, train_Y, resize)
    test_X, test_Y = remove_conflict(test_X, test_Y, resize)

    train_X = train_X[:max_size]
    train_Y = train_Y[:max_size]
    test_X = test_X[:max_size]
    test_Y = test_Y[:max_size]

    if binary:
        train_X = binary_img(train_X, 0.5)
        test_X = binary_img(test_X, 0.5)

    if encoding:
        train_X = encoding_img(train_X, encoding)
        test_X = encoding_img(test_X, encoding)

    train_dataset = Dataset(train_X, train_Y)
    test_dataset = Dataset(test_X, test_Y)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )
    return train_loader, test_loader


if __name__ == "__main__":
    RESIZE = (8, 8)

    EPOCH = 10  # 训练总轮数
    BATCH_SIZE = 32  # 一次迭代使用的样本数
    LR = 0.001  # 梯度下降的学习率
    SEED = 17  # 随机数种子
    ep_start = 0
    it_start = 0
    GRAYSCALE = 2

    set_seed(SEED)
    encoding = FRQI(GRAYSCALE)

    train_loader, test_loader = get_data_loader(RESIZE, encoding)

    # encoding = Qubit(16)
    # train_X = encoding_img(bin_train_X, encoding)
    # test_X = encoding_img(bin_test_X, encoding)
