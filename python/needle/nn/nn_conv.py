"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module
import math

class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        fan_in = in_channels * kernel_size * kernel_size
        fan_out = out_channels * kernel_size * kernel_size
        self.weight = Parameter(init.kaiming_uniform(fan_in, fan_out, (kernel_size, kernel_size, in_channels, out_channels), requires_grad=True, device=device, dtype=dtype))
        self.bias = None
        if bias:
            bound = 1.0 / (math.sqrt(fan_in))
            self.bias = Parameter(init.rand(out_channels, low=-bound, high=bound, requires_grad=True, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        x = x.transpose((1, 2)).transpose((2, 3))
        x = ops.conv(x, self.weight, self.stride, (self.kernel_size - 1) // 2)
        if self.bias is not None:
            bias = ops.broadcast_to(ops.reshape(self.bias, (1,1,1,self.out_channels)), x.shape)
            x = x + bias
        return x.transpose((2, 3)).transpose((1, 2))
        ### END YOUR SOLUTION