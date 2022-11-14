import cupy as cp
import torch



class Applygate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, gate):
        ctx.input = x
        # Enforce contiguous arrays to simplify RawKernel indexing.
        
        
        cupy_x = cupy.ascontiguousarray(cupy.from_dlpack(x.detach()))
        cupy_y = cupy.empty(cupy_x.shape, dtype=cupy_x.dtype)
        x_size = cupy_x.size
        bs = 128
        cupy_custom_kernel_fwd(
            (bs,), ((x_size + bs - 1) // bs,), (cupy_x, cupy_y, x_size)
        )
        # the ownership of the device memory backing cupy_y is implicitly
        # transferred to torch_y, so this operation is safe even after
        # going out of scope of this function.
        torch_y = torch.from_dlpack(cupy_y)
        
        return torch_y

    @staticmethod
    def backward(ctx, grad_y):
        # Enforce contiguous arrays to simplify RawKernel indexing.
        cupy_input = cupy.from_dlpack(ctx.input.detach()).ravel()
        cupy_grad_y = cupy.from_dlpack(grad_y.detach()).ravel()
        cupy_grad_x = cupy.zeros(cupy_grad_y.shape, dtype=cupy_grad_y.dtype)
        gy_size = cupy_grad_y.size
        bs = 128
        cupy_custom_kernel_bwd(
            (bs,),
            ((gy_size + bs - 1) // bs,),
            (cupy_input, cupy_grad_y, cupy_grad_x, gy_size),
        )
        # the ownership of the device memory backing cupy_grad_x is implicitly
        # transferred to torch_y, so this operation is safe even after
        # going out of scope of this function.
        torch_grad_x = torch.from_dlpack(cupy_grad_x)
        return torch_grad_x