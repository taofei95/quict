import torch
import time
# device = torch.device("cuda:0")
# dim = 65536
# a = torch.rand(dim, dim).to(device)
# b = torch.rand(dim, 1).to(device)
# s = time.time()
# c = torch.mm(a, b)
# print(time.time() - s)

# def f(x):
#     k = torch.tensor([[1., 2.]])
#     return torch.mm(k, x.T)

# c = torch.rand(2, 2, requires_grad=True)
# x = torch.tensor([[1., 0.]])
# y = f(x)
# loss = torch.sum(y)
# print(c.grad)
# loss.backward()
# print(c.grad)

x = torch.rand(1, requires_grad=True)
print(x)
y = x**2
z = x + x

z.backward()
print(x.grad)