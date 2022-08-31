import torch
from QuICT.qcda.mapping.ai.dataset import MappingGraphormerDataset


def test_graphormer_dataset():
    dataset = MappingGraphormerDataset()
    t_dataset, _ = dataset.split_tv()

    batch_size = 16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = t_dataset.loader(batch_size=batch_size, shuffle=True, device=device)
    
    x_batch, se_batch = next(iter(loader))
    assert x_batch.shape[0] == batch_size
    assert se_batch.shape[0] == batch_size
