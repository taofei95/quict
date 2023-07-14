import numpy as np
import random


class Dataset:
    """An abstract class representing a dataset."""

    def __init__(self, *datas: list):
        """Initialize a Dataset.
        
        This method accepts multiple lists of data and returns an iterable dataset.
        """
        assert all(
            len(datas[0]) == len(data) for data in datas
        ), "Length mismatch between datas"
        self._datas = datas

    def __getitem__(self, index):
        return tuple(data[index] for data in self._datas)

    def __setitem__(self, index, values):
        for data, value in zip(self._datas, values):
            data[index] = value

    def __len__(self):
        return len(self._datas[0])


class DataLoader:
    """Data loader, which supports iterable datasets."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        """Initialize a Dataloader.

        Args:
            dataset (Dataset): Dataset from which to load the data.
            batch_size (int, optional): The number of samples per batch to load. Defaults to 1.
            shuffle (bool, optional): Whether shuffle the data at every epoch. Defaults to True.
            drop_last (bool, optional): Whether to drop the last incomplete batch when the dataset size is not divisible by the batch size. Defaults to True.
        """
        self._dataset = dataset
        self._shuffle = shuffle
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
        if self._shuffle:
            random.shuffle(self._dataset)
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
