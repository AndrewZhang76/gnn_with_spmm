from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)

class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        return array_api.log(array_api.sum(array_api.exp(Z - Z.max(self.axes, keepdims=True).broadcast_to(Z.shape)), self.axes)) + Z.max(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]  # Input to the node during forward pass
        max_Z = Z.realize_cached_data().max(self.axes)
        new_shape = list(Z.shape)
        if self.axes is not None:
          if isinstance(self.axes, int):
            new_shape[self.axes] = 1
          else:
            for axis in self.axes:
              new_shape[axis] = 1
        else:
          for i in range(len(new_shape)):
            new_shape[i] = 1
        print(Z.shape, self.axes, new_shape)
        max_Z = max_Z.reshape(new_shape).broadcast_to(Z.shape)
        exp_Z = exp(Z - max_Z)  # Exponentiate the stabilized values
        sum_exp_z = summation(exp_Z, self.axes)
        grad_sum_exp_z = out_grad / sum_exp_z
        grad_exp_z = grad_sum_exp_z.reshape(new_shape).broadcast_to(Z.shape)
        return grad_exp_z * exp_Z
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)