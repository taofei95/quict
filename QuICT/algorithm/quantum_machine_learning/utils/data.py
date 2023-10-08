import numpy as np

from QuICT.tools.exception.algorithm import *


class Dataset:
    """An abstract class representing a map from keys to dataset.

    Accept multiple lists of data and returns an iterable dataset.

    Examples:
        >>> import numpy as np
        >>> from QuICT.algorithm.quantum_machine_learning.utils import Dataset, DataLoader
        >>> x = np.array([[2, 3], [0.3, 7], [-33, 1.2], [6, 5]])
        >>> y = np.array([0, 1, 1, 0])
        >>> dataset = Dataset(x, y)
        >>> len(dataset)
        4
        >>> dataset[0]
        (array([2., 3.]), 0)
    """

    def __init__(self, *datas: list):
        """Initialize a Dataset instance."""
        assert all(len(datas[0]) == len(data) for data in datas), DatasetError(
            "Length mismatch between datas."
        )
        self._datas = datas

    def __getitem__(self, index):
        """Get data according to index.

        Args:
            index: The index.

        Returns:
            tuple: Fetched data sample for the given index.
        """
        return tuple(np.array(data)[index] for data in self._datas)

    def __setitem__(self, index, values):
        """Set data values according to index."""
        for data, value in zip(self._datas, values):
            data[index] = value

    def __len__(self):
        """Get the length of dataset."""
        return len(self._datas[0])


class DataLoader:
    """Data loader, which provides an iterable over the given dataset.

    Args:
        dataset (Dataset): Dataset from which to load the data.
        batch_size (int, optional): The number of samples per batch to load. Defaults to 1.
        shuffle (bool, optional): Whether shuffle the data at every epoch. Defaults to True.
        drop_last (bool, optional): Whether to drop the last incomplete batch when the dataset size is not divisible
            by the batch size. Defaults to True.

    Examples:
        >>> import numpy as np
        >>> from QuICT.algorithm.quantum_machine_learning.utils import Dataset, DataLoader
        >>> x = np.array([[2, 3], [0.3, 7], [-33, 1.2], [6, 5]])
        >>> y = np.array([0, 1, 1, 0])
        >>> dataset = Dataset(x, y)
        >>> len(dataset)
        4
        >>> dataset[0]
        (array([2., 3.]), 0)
        >>> dataloader = DataLoader(dataset=dataset, batch_size=2)
        >>> len(dataloader)
        2
        >>> for data in dataloader:
        >>>     print(data)
        (array([[6. , 5. ],
                [0.3, 7. ]]), array([0, 1]))
        (array([[  2. ,   3. ],
                [-33. ,   1.2]]), array([0, 1]))
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        """Initialize a Dataloader instance."""
        self._dataset = dataset
        self._shuffle = shuffle
        self._batch_size = batch_size
        self._drop_last = drop_last

        self._it = 0
        self._end = self.__len__()
        self._idx = np.arange(len(self._dataset))

    def __len__(self) -> int:
        """Get the length of the dataloader."""
        length = len(self._dataset)
        length = (
            length // self._batch_size
            if self._drop_last
            else np.ceil(length / self._batch_size)
        )
        return length

    def __iter__(self):
        """Return an iterator of samples in the dataset."""
        self._it = 0
        if self._shuffle:
            self._idx = np.arange(len(self._dataset))
            np.random.shuffle(self._idx)
        return self

    def __next__(self):
        """Access the next element of the dataloader."""
        if self._it < self._end:
            ret_data = self._dataset[
                self._idx[
                    self._it * self._batch_size : (self._it + 1) * self._batch_size
                ]
            ]
            self._it += 1
            return ret_data
        else:
            raise StopIteration
