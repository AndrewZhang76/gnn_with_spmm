from needle.autograd import Tensor
from numpy import divide
from .nn_basic import Parameter, Module, ReLU
from needle import ops
import needle.init as init

class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True, device = None, dtype="float32"):
        super().__init__()
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))
        self.bias = Parameter(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype).reshape((1, out_features))) if bias else None

    def forward(self, X: Tensor, A: Tensor) -> Tensor:
        out = ops.matmul(A, X)
        out = ops.matmul(out, self.weight)
        if self.bias is not None:
            out = ops.add(out, self.bias.broadcast_to(out.shape))
        return out