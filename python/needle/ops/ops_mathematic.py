"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a**b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs[0], node.inputs[1]
        return out_grad *b*(a**(b-1)), out_grad *(a**b)*log(a)

        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, = node.inputs
        return out_grad * self.scalar * power_scalar(a, self.scalar-1)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a/b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        return out_grad/b, -out_grad*a/b**2
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a/self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad/self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes:
          index = list(range(len(a.shape)))
          index[self.axes[0]], index[self.axes[1]] = index[self.axes[1]], index[self.axes[0]]
          return a.permute(tuple(index))
        else:
          index = list(range(len(a.shape)))
          index[-1], index[-2] = index[-2], index[-1]
          return a.permute(tuple(index))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, axes=self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.compact().reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape).compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        input_shape = (1,) * (len(out_grad.shape) - len(input_shape)) + input_shape
        reduce_axes = [i for i in range(len(out_grad.shape)) if input_shape[i] == 1 and out_grad.shape[i]!= 1]
        out_grad = summation(out_grad, axes = tuple(reduce_axes))
        out_grad = out_grad.reshape(node.inputs[0].shape)
        return out_grad
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            return a.sum(axis = None)
        elif isinstance(self.axes, int):
            return a.sum(self.axes)
        else:
            for axis in sorted(self.axes, reverse=True):
                a = a.sum(axis = axis)
            return a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a_shape = node.inputs[0].shape
        if self.axes is None:
            # Sum over all axes
            return broadcast_to(out_grad, a_shape)
        else:
            axes = self.axes
            if isinstance(axes, int):
                axes = (axes,)
            # Adjust negative axes
            axes = tuple([axis if axis >= 0 else axis + len(a_shape) for axis in axes])
            # Reshape out_grad to add singleton dimensions at summed axes
            shape = list(out_grad.shape)
            for axis in sorted(axes):
                shape.insert(axis, 1)
            out_grad_reshaped = reshape(out_grad, shape)
            # Broadcast to input shape
            return broadcast_to(out_grad_reshaped, a_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        if len(a.shape) == len(b.shape):
          grad_a = matmul(out_grad, transpose(b, axes=(-1, -2)))
          grad_b = matmul(transpose(a, axes=(-1, -2)), out_grad)
          return grad_a, grad_b
        elif len(a.shape) > len(b.shape):
          out = matmul(transpose(a),out_grad)
          for _ in range(len(a.shape) - len(b.shape)):
              out = summation(out, 0)
          return matmul(out_grad, transpose(b)), out
        else:
          out = matmul(out_grad, transpose(b))
          for _ in range(len(b.shape) - len(a.shape)):
              out = summation(out, 0)
          return out, matmul(transpose(a), out_grad)
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * (exp(node.inputs[0]))
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0].realize_cached_data()
        return out_grad * Tensor(a > 0, device=out_grad.device)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad * mul_scalar(power_scalar(exp(-a)+exp(a), -2), 4)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        stacked_shape = list(args[0].shape)
        stacked_shape.insert(self.axis, len(args))
        stacked_array = array_api.empty(shape=stacked_shape, device=args[0].device)
        
        for i, tensor in enumerate(args):
            slices = [slice(None)] * len(stacked_shape)
            slices[self.axis] = i
            stacked_array[tuple(slices)] = tensor
        
        return stacked_array
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        num_splits = A.shape[self.axis]
        # Initialize a list to hold each split
        split_tensors = []
        
        # Create slices for each split along the specified axis
        for i in range(num_splits):
            # Create a slicing object for each dimension
            slices = [slice(None)] * len(A.shape)  # Initialize to take all elements along other dimensions
            slices[self.axis] = slice(i, i + 1)  # Take only the current slice along the specified axis
            # Slice A and remove the dimension we split along to match the output shape of split parts
            split_tensors.append(A[tuple(slices)].sum(axis=self.axis))
        
        # Return the list as a tuple
        return tuple(split_tensors)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.flip(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        dilated_shape = list(a.shape)
        for axis in self.axes:
          if axis < len(dilated_shape):
            dilated_shape[axis] += dilated_shape[axis] * self.dilation
        dialted_arr = array_api.full(tuple(dilated_shape), 0, device=a.device)
        dialted_arr.fill(0)
        orig_slices = [slice(0, dim) for dim in dilated_shape]
        for axis in self.axes:
          if axis < len(dilated_shape):
            orig_slices[axis] = slice(0, dilated_shape[axis], self.dilation + 1)
        dialted_arr[tuple(orig_slices)] = a
        return dialted_arr
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        orig_slices = [slice(0, dim) for dim in a.shape]
        for axis in self.axes:
          if axis < len(orig_slices):
            orig_slices[axis] = slice(0, a.shape[axis], self.dilation + 1)
        return a[tuple(orig_slices)]
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        A_padded = A.pad(((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        N, H, W, C_in = A_padded.shape
        K, H_B, W_B, C_out = B.shape
        Ns, Hs, Ws, Cs = A_padded.strides
        # Calculate output dimensions
        H_out = (H - K + 1) // self.stride
        W_out = (W - K + 1) // self.stride
        out_shape = (N, H_out, W_out, K, K, C_in)
        out_strides = (Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs)
        A_strided = A_padded.as_strided(shape=out_shape, strides=out_strides).compact().reshape((N * H_out * W_out, K * C_in * K))
        out = (A_strided @ B.compact().reshape((K * H_B * W_B, C_out)))
        return out.compact().reshape((N, H_out, W_out, C_out))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        A, B = node.inputs
        
        # Gradient with respect to A (input)
        B_flipped = flip(B, (0, 1))  # Flip the kernel
        
        # Dilate out_grad if stride > 1
        if self.stride > 1:
            out_grad_dilated = dilate(out_grad, (1, 2), self.stride - 1)
        else:
            out_grad_dilated = out_grad
        
        # Calculate padding for input gradient
        padding_A = B.shape[0] - self.padding - 1
        B_transposed = transpose(B_flipped, (2,3))
        A_grad = conv(out_grad_dilated, B_transposed, padding=padding_A, stride=1)

        # Gradient with respect to B (weights)
        
        # Transpose A and out_grad for weight gradient calculation
        A_transposed = A.transpose((0,3))
        out_grad_transposed = out_grad_dilated.transpose((0,1)).transpose((1,2))
        
        B_grad = conv(A_transposed, out_grad_transposed, padding=self.padding, stride=1)
        
        # Transpose B_grad back to original shape
        B_grad = B_grad.transpose((0,1)).transpose((1,2))
      
        return A_grad, B_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


