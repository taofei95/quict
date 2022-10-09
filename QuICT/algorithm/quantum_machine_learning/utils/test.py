import torch
import numpy as np


def gatef(x):
    gate = torch.ones([2, 2], dtype=torch.float64)
    gate[0, 0] = torch.sin(x)
    gate[1, 1] = torch.cos(x)
    return gate


param = torch.nn.Parameter(torch.rand(1, dtype=torch.float64))
gate = gatef(param)
optim = torch.optim.Adam(params=[param], lr=0.05)
state = torch.tensor([1, 0], dtype=torch.float64)
state = state.reshape(2, 1)

for i in range(10):
    optim.zero_grad()
    new_state = torch.mm(gate, state)
    loss = abs(new_state - torch.tensor([[0.5, 1]], dtype=torch.float64)).mean()
    loss.backward()
    print(param.grad)
    optim.step()

