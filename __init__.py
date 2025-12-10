import torch
import selective_scan_cuda

def selective_scan_forward(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False):
    return selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus)

def selective_scan_backward(u, delta, A, B, C, D, z, delta_bias, dout, x, out, dz, delta_softplus, recompute_out_z):
    return selective_scan_cuda.bwd(u, delta, A, B, C, D, z, delta_bias, dout, x, out, dz, delta_softplus, recompute_out_z)

class SelectiveScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False):
        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus)
        ctx.delta_softplus = delta_softplus
        ctx.has_z = z is not None
        ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, x, out, *rest)
        return out if not ctx.has_z else (out, rest[0])
    
    @staticmethod
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, z, delta_bias, x, out, *rest = ctx.saved_tensors
        dz = args[0] if ctx.has_z and len(args) > 0 else None
        out_z = rest[0] if ctx.has_z and len(rest) > 0 else None
        dout = dout.contiguous()
        if dz is not None:
            dz = dz.contiguous()
        grads = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, z, delta_bias,
            dout, x, out, dz, ctx.delta_softplus, False
        )
        return tuple(grads) + (None,) * (9 - len(grads))

def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False):
    return SelectiveScanFunction.apply(u, delta, A, B, C, D, z, delta_bias, delta_softplus)

__version__ = '0.1.0'
__all__ = ['selective_scan_fn', 'selective_scan_forward', 'selective_scan_backward']

