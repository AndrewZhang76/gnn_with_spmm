import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset
import os
import pickle
from backend_ndarray.ndarray import *
import numpy as np
import dgl
import torch
import dgl.data
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import networkx as nx

class GNNDataset(Dataset):
    def __init__(
        self,
        train: bool,
    ):
        ### BEGIN YOUR SOLUTION
        self.train = train
        
        dataset = dgl.data.GINDataset('PROTEINS', self_loop=True)

        num_examples = len(dataset)
        num_train = int(num_examples * 0.8)
        if self.train:
            dataset = dataset[:num_train]
        else:
            dataset = dataset[num_train:]
        self.X = []
        self.y = []
        device = cpu()
        for data in dataset:
            graph = nx.to_numpy_array(data[0].to_networkx())
            graph_dense = NDArray(graph).to(device=device)
            graph_sparse = graph_dense.to_sparse()
            y = data[1].numpy()
            y_ndarray = NDArray(y).to(device=device)
            self.X.append(graph_sparse)
            self.y.append(y_ndarray)
    
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        """
        ### BEGIN YOUR SOLUTION
        X_item = self.X[index]
        y_item = self.y[index]

        return X_item, y_item
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.X)
        ### END YOUR SOLUTION
