"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype = dtype))
        self.bias = Parameter(init.kaiming_uniform(out_features, 1, device=device, dtype = dtype).reshape((1, out_features))) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        y = ops.matmul(X, self.weight)
        if self.bias:
          return ops.add(y, self.bias.broadcast_to(y.shape))
        else:
          return y
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        size = 1
        for i in range(len(X.shape)):
          if i > 0:
            size *= X.shape[i]
        return ops.reshape(X, (X.shape[0], size))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = x
        for module in self.modules:
          out = module.forward(out)
        return out
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        n, out_d = logits.shape[0], logits.shape[1]
        one_hot =init.one_hot(out_d, y, device=logits.device, dtype =logits.dtype)
        log_sum_exp_logits = ops.logsumexp(logits, axes = tuple([1]))
        gt = ops.summation(ops.multiply(logits, one_hot), axes=1)
        loss = ops.summation(log_sum_exp_logits-gt)
        return ops.divide_scalar(loss, n)
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.weight = Parameter(init.ones(dim, device=device), requires_grad=True)
        self.bias = Parameter(init.zeros(dim, device=device), requires_grad=True)

        # Running statistics (buffers)
        self.running_mean = init.zeros(dim, device=device)
        self.running_var = init.ones(dim, device=device)
        

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mean = x.sum(axes=(0,)) / x.shape[0]
            variance = ((x - mean.broadcast_to(x.shape)) ** 2).sum(axes=0) / x.shape[0]
            # Step 2: Normalize the input
            x_normalized = (x -  mean.broadcast_to(x.shape)) / ((variance + self.eps) ** 0.5).broadcast_to(x.shape)
            # Update running statistics using momentum
            self.running_mean = self.momentum * mean.data + (1 - self.momentum) * self.running_mean
            self.running_var = self.momentum * variance.data + (1 - self.momentum) * self.running_var
            return self.weight.broadcast_to(x.shape) * x_normalized + self.bias.broadcast_to(x.shape)
        else:
            # Use running statistics for inference
            x_normalized = (x - self.running_mean) / (self.running_var + self.eps) ** 0.5
            return self.weight.broadcast_to(x.shape) * x_normalized + self.bias.broadcast_to(x.shape)

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))

class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device), requires_grad = True)
        self.bias = Parameter(init.zeros(dim, device=device), requires_grad = True)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        mean = x.sum(axes=1).reshape((x.shape[0],1)) / x.shape[1]
        variance = ((x - mean.broadcast_to(x.shape)) ** 2).sum(axes=1).reshape((x.shape[0],1)) / x.shape[1]
        # Step 2: Normalize the input
        x_normalized = (x -  mean.broadcast_to(x.shape)) / ((variance + self.eps) ** 0.5).broadcast_to(x.shape)

        # Step 3: Scale and shift with learned parameters
        output = self.weight.broadcast_to(x.shape) * x_normalized + self.bias.broadcast_to(x.shape)

        return output
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
          return x * (init.randb(*x.shape, p=(1-self.p), device=x.device, dtype=x.dtype)) / (1-self.p)
        else:
          return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
