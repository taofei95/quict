from os import path as osp
import os
from typing import Dict, List, Tuple
from torch_geometric.data import HeteroData, Data, Batch
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset


class MappingDataSet(Dataset):
    def __init__(self, data_dir: str, device: str) -> None:
        super().__init__()
        self._circ_cnt = 0
        self._data_dir = data_dir
        self.device = device
        if not osp.exists(data_dir):
            raise FileNotFoundError(data_dir)
        print(f"Loading dataset from {data_dir}")
        for _, _, filenames in os.walk(data_dir):
            self._circ_cnt += len(filenames)

    def __len__(self):
        return self._circ_cnt

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Data], int]:
        f_path = osp.join(self._data_dir, f"{idx}.pt")
        with open(f_path, "rb") as f:
            circ_dict_raw, target = torch.load(f)
            circ_dict = {}
            circ_dict["lc"] = circ_dict_raw["lc"].to(self.device)
            circ_dict["topo"] = circ_dict_raw["topo"].to(self.device)
            return circ_dict, target


class MappingDataLoaderFactory:
    @staticmethod
    def get_loader(
        batch_size: int, shuffle: bool, device: str, data_dir: str = None,
    ):
        if data_dir is None:
            data_dir = osp.dirname(osp.realpath(__file__))
            data_dir = osp.join(data_dir, "data")
            data_dir = osp.join(data_dir, "processed")

        dataset = MappingDataSet(data_dir=data_dir, device=device)

        def collat_fn(batch):
            circ_dicts_raw, targets = zip(*batch)
            topo_batch = []
            lc_batch = []
            for d in circ_dicts_raw:
                topo_batch.append(d["topo"])
                lc_batch.append(d["lc"])
            circ_dicts = {}
            circ_dicts["topo"] = Batch.from_data_list(topo_batch)
            circ_dicts["lc"] = Batch.from_data_list(lc_batch)
            targets = torch.tensor(list(targets)).to(device)
            return circ_dicts, targets

        loader = TorchDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collat_fn,
        )
        return loader


if __name__ == "__main__":
    loader = MappingDataLoaderFactory.get_loader(
        batch_size=1, shuffle=False, device="cpu"
    )
    for batch in loader:
        print(batch)
        break
