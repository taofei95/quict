import torch
from QuICT.qcda.mapping.ai.circuit_transformer import *


def test_self_attn():
    for feat_dim in [3, 4, 5, 6, 7]:
        model = SelfAttn(feat_dim)
        for node_num in [1, 3, 20]:
            x_no_batch = torch.randn(node_num, feat_dim)
            attn_bias = torch.randn(node_num, node_num)
            with torch.no_grad():
                y_no_batch = model(x_no_batch)
                assert y_no_batch.shape == x_no_batch.shape
                y_no_batch = model(x_no_batch, attn_bias)
                assert y_no_batch.shape == x_no_batch.shape

            for batch_size in [1, 3, 5]:
                x_batch = torch.randn(batch_size, node_num, feat_dim)
                with torch.no_grad():
                    y_batch = model(x_batch)
                    assert y_batch.shape == x_batch.shape
                    y_batch = model(x_batch, attn_bias)
                    assert y_batch.shape == x_batch.shape


def test_multi_head_attn():
    for head in [1, 3, 5]:
        for feat_dim in [3, 4, 5, 6, 7]:
            model = MultiHeadAttn(head, feat_dim)
            for node_num in [1, 3, 20]:
                x_no_batch = torch.randn(node_num, feat_dim)
                attn_bias = torch.randn(node_num, node_num)
                with torch.no_grad():
                    y_no_batch = model(x_no_batch)
                    assert y_no_batch.shape == x_no_batch.shape
                    y_no_batch = model(x_no_batch, attn_bias)
                    assert y_no_batch.shape == x_no_batch.shape

                for batch_size in [1, 3, 5]:
                    x_batch = torch.randn(batch_size, node_num, feat_dim)
                    with torch.no_grad():
                        y_batch = model(x_batch)
                        assert y_batch.shape == x_batch.shape
                        y_batch = model(x_batch, attn_bias)
                        assert y_batch.shape == x_batch.shape


if __name__ == "__main__":
    import pytest
    import os

    pytest.main([os.path.abspath(__file__)])
