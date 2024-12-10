from needle.autograd import Tensor
from .nn_basic import Module, Parameter, ReLU
from needle import ops
import needle.init as init

class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))
        self.bias = Parameter(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype).reshape((1, out_features))) if bias else None

    def forward(self, X: Tensor, A: Tensor) -> Tensor:
        out = ops.matmul(A, X)
        out = ops.matmul(out, self.weight)
        if self.bias is not None:
            out = out + self.bias.broadcast_to(out.shape)
        return out

class GCN(Module):
    def __init__(self, in_features, hidden_features, out_features, device=None, dtype="float32"):
        super().__init__()
        self.gcn1 = GraphConvolution(in_features, hidden_features, device=device, dtype=dtype)
        self.relu = ReLU()
        self.gcn2 = GraphConvolution(hidden_features, out_features, device=device, dtype=dtype)

    def forward(self, X: Tensor, A: Tensor):
        h = self.gcn1(X, A)
        h = self.relu(h)
        h = self.gcn2(h, A)
        return h