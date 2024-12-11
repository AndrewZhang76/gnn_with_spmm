import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)

class GCN(ndl.nn.Module):
    def __init__(self, in_features, hidden_features, out_features, device=None, dtype="float32"):
        super().__init__()
        self.gcn1 = nn.GraphConvolution(in_features, hidden_features, device=device, dtype=dtype)
        self.relu = nn.ReLU()
        self.gcn2 = nn.GraphConvolution(hidden_features, out_features, device=device, dtype=dtype)

    def forward(self, X: ndl.Tensor, A: ndl.Tensor):
        h = self.gcn1(X, A)
        h = self.relu(h)
        h = self.gcn2(h, A)
        return h