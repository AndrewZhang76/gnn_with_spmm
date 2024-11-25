import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION



def kaiming_uniform(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    if shape is None:
        # If shape is provided, use it directly for initialization
        bound = math.sqrt(6.0 / fan_in)  
        return rand(fan_in, fan_out, low = -bound, high = bound, **kwargs)
    else:
        # Otherwise, use fan_in and fan_out to calculate the bound
        bound = math.sqrt(6.0 / fan_in)
        return rand(*shape, low = -bound, high = bound, **kwargs)
    ### END YOUR SOLUTION

def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION