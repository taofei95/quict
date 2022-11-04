import collections
import torch
from torchvision import datasets, transforms
import numpy as np


class MNISTDataset:
    def __init__(self, root="./data/", download=True, device=torch.device("cuda:0")):
        data_train = datasets.MNIST(root=root, train=True, download=download)
        data_test = datasets.MNIST(root=root, train=False, download=download)
        self._device = device
        self._x_train = data_train.data.to(device)
        self._y_train = data_train.targets.to(device)
        self._x_test = data_test.data.to(device)
        self._y_test = data_test.targets.to(device)

    @property
    def x_train(self):
        return self._x_train

    @property
    def y_train(self):
        return self._y_train

    @property
    def x_test(self):
        return self._x_test

    @property
    def y_test(self):
        return self._y_test

    def filter_targets(self, class0=3, class1=6):
        idx_train = (self._y_train == class0) | (self._y_train == class1)
        self._x_train, self._y_train = (
            self._x_train[idx_train],
            self._y_train[idx_train],
        )
        self._y_train = self._y_train == class1

        idx_test = (self._y_test == class0) | (self._y_test == class1)
        self._x_test, self._y_test = self._x_test[idx_test], self._y_test[idx_test]
        self._y_test = self._y_test == class1

    def downscale(self, resize=(4, 4)):
        transform = transforms.Resize(size=resize)
        self._x_train = transform(self._x_train) / 255.0
        self._x_test = transform(self._x_test) / 255.0

    def remove_conflict(self, resize=(4, 4)):
        def rmcon(X, Y):
            x_dict = collections.defaultdict(set)
            for x, y in zip(X, Y):
                x_dict[tuple(x.cpu().detach().numpy().flatten())].add(y.item())
            X_rmcon = []
            Y_rmcon = []
            for x in x_dict.keys():
                if len(x_dict[x]) == 1:
                    X_rmcon.append(np.array(x).reshape(resize))
                    Y_rmcon.append(list(x_dict[x])[0])
            X_rmcon = torch.from_numpy(np.array(X_rmcon)).to(self._device)
            Y_rmcon = torch.from_numpy(np.array(Y_rmcon)).to(self._device)
            return X_rmcon, Y_rmcon

        self._x_train, self._y_train = rmcon(self._x_train, self._y_train)
        self._x_test, self._y_test = rmcon(self._x_test, self._y_test)

    def binary_img(self, threshold=0.5):
        self._x_train = self._x_train > threshold
        self._x_test = self._x_test > threshold
