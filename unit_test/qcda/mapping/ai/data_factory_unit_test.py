import torch
from QuICT.qcda.mapping.ai.data_factory import DataFactory


def test_data_gen():
    qubit_num, layer_num = 30, 30
    factory = DataFactory(
        max_qubit_num=qubit_num, max_layer_num=layer_num
    )
    node_num = qubit_num * (layer_num + 1)

    # Run multiple times due to randomness.
    for _ in range(5):
        _, _, x, _, _ = factory.get_one()

        assert x.shape == torch.Size((node_num,))
