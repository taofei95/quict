class Dataset:
    def __init__(self, *datas: list):
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
