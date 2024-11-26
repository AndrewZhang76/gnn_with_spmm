import needle as ndl

import needle.nn as nn
from needle import ops

class GNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        h = self.linear(x)
        h = self.relu(h)

        ##TODO: Implement spmm
        ##Suppose to be spmm
        #h = ndl.spmm(adj, h)

        ##baseline, direct matrix multiplication
        h = ops.matmul(h, adj)
        return h

class GNN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(GNN, self).__init__()
        self.layer1 = GNNLayer(in_features, hidden_features)
        self.layer2 = GNNLayer(hidden_features, out_features)

    def forward(self, x, adj):
        h = self.layer1(x, adj)
        h = self.layer2(h, adj)
        return h