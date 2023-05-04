import numpy as np
import random

from .dataset import Dataset


class DataLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        self._dataset = dataset
        if shuffle:
            random.shuffle(self._dataset)
        self._batch_size = batch_size
        self._drop_last = drop_last

        self._it = 0
        self._end = self.__len__()

    def __len__(self) -> int:
        length = len(self._dataset)
        length = (
            length // self._batch_size
            if self._drop_last
            else np.ceil(length / self._batch_size)
        )
        return length

    def __iter__(self):
        self._it = 0
        return self

    def __next__(self):
        if self._it < self._end:
            ret_data = self._dataset[
                self._it * self._batch_size : (self._it + 1) * self._batch_size
            ]
            self._it += 1
            return ret_data
        else:
            raise StopIteration
