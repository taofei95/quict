import torch
import torch.nn as nn
import cupy as cp

class LinearFunction(torch.autograd.Function):
    # 必须是staticmethod
    @staticmethod
    # 第一个是ctx，第二个是input，其他是可选参数。
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())  # torch.t()方法，对2D tensor进行转置
        if bias is not None:
            # expand_as(tensor)等价于expand(tensor.size()), 将原tensor按照新的size进行扩展
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output): 
        # grad_output为反向传播上一级计算得到的梯度值
        # input, weight, bias = ctx.saved_variables
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        # 分别代表输入,权值,偏置三者的梯度
        # 判断三者对应的Variable是否需要进行反向求导计算梯度
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias


class Linear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(Linear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)
        self.weight.data.uniform_(-0.1, 0.1)
        if bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)
    
    def f(self, y):
        # y_gpu = y.to(torch.device("cuda:0"))
        # y_cupy = cp.from_dlpack(y_gpu.detach()) 
        # y_tensor = torch.from_dlpack(y_cupy).cpu()
        # y[:] = y_tensor
        y_detach = y.detach()
        y[:] = y_detach
        return y
    
    def forward(self, input):
        y_pred = LinearFunction.apply(input, self.weight, self.bias)
        # y_pred = self.f(y_pred)
        return y_pred


model = Linear(input_features=4, output_features=3)
optim = torch.optim.Adam(([dict(params=model.parameters(), lr=0.1)]))
x= torch.rand((2, 4), requires_grad=True)
for i in range(10):
    optim.zero_grad()
    y = model(x)
    loss = torch.sum(y)
    loss.backward()
    print(x.grad)
    # print(loss.item())
    optim.step()