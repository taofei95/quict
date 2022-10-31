import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class MNIST_Classifier:
    def __init__(self, target1=3, target2=6, resize=(4, 4)):
        self.target1 = target1
        self.target2 = target2
        self.resize = transforms.Resize(size=resize)
        self.load_data()

    def load_data(self):
        data_train = datasets.MNIST(root="./data/", train=True)
        data_test = datasets.MNIST(root="./data/", train=False)
        x_train = data_train.data
        y_train = data_train.targets
        x_test = data_test.data
        y_test = data_test.targets

        # Data preprocessing
        # Keep target1 and target2
        x_train, y_train = self._filter_targets(x_train, y_train)
        x_test, y_test = self._filter_targets(x_test, y_test)
        # Downscale
        x_train = self.resize(x_train) / 255.0
        x_test = self.resize(x_test) / 255.0
        # Binary images
        x_train = torch.tensor(x_train > 0.1, dtype=torch.float32)
        x_test = torch.tensor(x_test > 0.1, dtype=torch.float32)
        # Remove ambiguous data
        x_train, y_train = self._remove_confict(x_train, y_train)
        # x_test, y_test = self._remove_confict(x_test, y_test)

    def _filter_targets(self, x, y):
        idx = (y == self.target1) | (y == self.target2)
        x, y = x[idx], y[idx]
        y = y == self.target1
        return x, y

    def _remove_confict(self, X, Y):
        img_dict = {}
        # for x, y in zip(X, Y):
        plt.imshow(X[0])
        plt.show()


c = MNIST_Classifier()

