import collections
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np


# class MNISTDataset(Dataset):
#     def __init__(self, root="./data/", train: bool = True, batch_size=32, device=torch.device("cuda:0")):
#         super(Dataset, self).__init__()
#         train_data = datasets.MNIST(root=root, train=True, download=True)
#         test_data = datasets.MNIST(root=root, train=False, download=True)
#         self._device = device
#         self._x_train = train_data.data.to(device)
#         self._y_train = train_data.targets.to(device)
#         self._x_test = test_data.data.to(device)
#         self._y_test = test_data.targets.to(device)
#         # train_loader = data.DataLoader(
#         #     dataset=train_data, batch_size=batch_size, shuffle=True
#         # )
#         # test_loader = data.DataLoader(
#         #     dataset=test_data, batch_size=batch_size, shuffle=True
#         # )
#         # for i, item in enumerate(train_loader):
#         #     img, label = item
#         #     print(img)
class MNISTDataset(Dataset):
    def __init__(
        self, root="./data/", train: bool = True, device=torch.device("cuda:0"),
    ):
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
        idx = (self._y == class0) | (self._y == class1)
        self._x, self._y = (self._x[idx], self._y[idx])
        self._y = self._y == class1

    def downscale(self, resize=(4, 4)):
        transform = transforms.Resize(size=resize)
        self._x = transform(self._x) / 255.0

    def remove_conflict(self, resize=(4, 4)):
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
        self._x = self._x > threshold


# dataset = MNISTDataset(train=True)
# train_loader = data.DataLoader(dataset=dataset, batch_size=32, shuffle=True)

# import tqdm

# loader = tqdm.tqdm(train_loader)
# for batch in loader:
#     print(batch[0].shape)
