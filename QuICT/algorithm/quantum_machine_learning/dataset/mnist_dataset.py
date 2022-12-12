import collections
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np


class MNISTDataset(Dataset):
    """The dataloader of MNIST dataset."""

    def __init__(
        self,
        root="./data/",
        train: bool = True,
        device=torch.device("cuda:0"),
    ):
        """The MNIST Dataset.

        Args:
            root (str, optional): The root directory of the dataset. Defaults to "./data/".
            train (bool, optional): Load the training set or testing set. Defaults to True.
            device (torch.device, optional): The device to which the model is assigned.
                Defaults to "cuda:0".
        """
        super(Dataset, self).__init__()
        data = datasets.MNIST(root=root, train=train, download=True)
        self._device = device
        self._x = data.data.to(device)
        self._y = data.targets.to(device)

    def __len__(self):
        return self._y.shape[0]

    def __getitem__(self, index):
        return self._x[index], self._y[index]

    def filter_targets(self, class0=3, class1=6):
        """Filter the dataset to keep only data belonging to two classes.

        Args:
            class0 (int, optional): Class marked as 0. Defaults to 3.
            class1 (int, optional): Class marked as 1. Defaults to 6.
        """
        idx = (self._y == class0) | (self._y == class1)
        self._x, self._y = (self._x[idx], self._y[idx])
        self._y = self._y == class1

    def downscale(self, resize=(4, 4)):
        """Downscale the images for classification.

        Args:
            resize (tuple, optional): The size of the downscaled image. Defaults to (4, 4).
        """
        transform = transforms.Resize(size=resize)
        self._x = transform(self._x) / 255.0

    def remove_conflict(self, resize=(4, 4)):
        """Remove conflict images that are labeled as belonging to both classes.

        Args:
            resize (tuple, optional): The size of the downscaled image. Defaults to (4, 4).
        """
        x_dict = collections.defaultdict(set)
        for x, y in zip(self._x, self._y):
            x_dict[tuple(x.cpu().detach().numpy().flatten())].add(y.item())
        X_rmcon = []
        Y_rmcon = []
        for x in x_dict.keys():
            if len(x_dict[x]) == 1:
                X_rmcon.append(np.array(x).reshape(resize))
                Y_rmcon.append(list(x_dict[x])[0])
        self._x = torch.from_numpy(np.array(X_rmcon)).to(self._device)
        self._y = torch.from_numpy(np.array(Y_rmcon)).to(self._device)

    def binary_img(self, threshold=0.5):
        """Binarize the image.

        Args:
            threshold (float, optional): Threshold for image binarization.
                Pixels greater than threshold are regarded as 1, otherwise are regarded as 0. Defaults to 0.5.
        """
        self._x = (self._x > threshold).type(torch.float32)
