import collections
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


class MNISTDataset:
    def __init__(
        self, class0=3, class1=6, resize=(4, 4), threshold=0.5,
    ):
        self.class0 = class0
        self.class1 = class1
        self.resize = resize
        self.threshold = threshold

    def load_data(self):
        data_train = datasets.MNIST(root="./data/", train=True)
        data_test = datasets.MNIST(root="./data/", train=False)
        x_train = data_train.data
        y_train = data_train.targets
        x_test = data_test.data
        y_test = data_test.targets
        return x_train, y_train, x_test, y_test

        # # Data preprocessing
        # # Keep class0 and class1
        # x_train, y_train = self._filter_targets(x_train, y_train)
        # x_test, y_test = self._filter_targets(x_test, y_test)
        # # Downscale
        # x_train = self._downscale(x_train)
        # x_test = self._downscale(x_test)
        # # Remove ambiguous data
        # x_train, self.y_train = self._remove_conflict(x_train, y_train)
        # x_test, self.y_test = self._remove_conflict(x_test, y_test)
        # # Binary images
        # self.x_train = self._binary_img(x_train)
        # self.x_test = self._binary_img(x_test)

    def filter_targets(self, x, y):
        idx = (y == self.class0) | (y == self.class1)
        x, y = x[idx], y[idx]
        y = y == self.class1
        return x, y

    def downscale(self, x):
        transform = transforms.Resize(size=self.resize)
        return transform(x) / 255.0

    def remove_conflict(self, X, Y):
        x_dict = collections.defaultdict(set)
        for x, y in zip(X, Y):
            x_dict[tuple(x.numpy().flatten())].add(y.item())
        X_rmcon = []
        Y_rmcon = []
        for x in x_dict.keys():
            if len(x_dict[x]) == 1:
                X_rmcon.append(np.array(x).reshape(self.resize))
                Y_rmcon.append(list(x_dict[x])[0])
        X_rmcon = torch.from_numpy(np.array(X_rmcon))
        Y_rmcon = torch.from_numpy(np.array(Y_rmcon))
        return X_rmcon, Y_rmcon

    def binary_img(self, x):
        return (x > self.threshold).clone()
