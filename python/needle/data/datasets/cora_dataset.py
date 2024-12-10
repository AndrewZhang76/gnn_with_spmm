import numpy as np
import scipy.sparse as sp
from ..data_basic import Dataset
from ..data_transformers import ToTensor, Compose
import needle as ndl

class CoraDataset(Dataset):
    def __init__(self, feature_path, adj_path, label_path,
                 idx_train=None, idx_val=None, idx_test=None,
                 device=ndl.cpu(), transforms=None):
        self.X = np.load(feature_path)   # shape: [num_nodes, num_features]
        self.y = np.load(label_path)     # shape: [num_nodes]
        self.A = sp.load_npz(adj_path)   # adjacency matrix in sparse format

        # Add self-loops
        self.A = self.A + sp.eye(self.A.shape[0])
        deg = np.array(self.A.sum(1)).flatten()
        deg_inv_sqrt = 1.0 / np.sqrt(deg)
        D_inv_sqrt = sp.diags(deg_inv_sqrt)
        self.A = D_inv_sqrt @ self.A @ D_inv_sqrt
        
        # Define train/val/test indices if not provided
        self.idx_train = idx_train if idx_train is not None else np.arange(0, 140)
        self.idx_val = idx_val if idx_val is not None else np.arange(140, 640)
        self.idx_test = idx_test if idx_test is not None else np.arange(1708, 2708)

        # Default transform: ToTensor
        if transforms is None:
            transforms = Compose([ToTensor(device=device)])
        self.transforms = transforms

    def __len__(self):
        # Only one graph
        return 1

    def __getitem__(self, idx):
        # Return the entire graph
        # Convert A to dense if sparse matmul isn't supported
        A = self.A.toarray()
        data = (self.X, self.y, A)
        X, y, A = self.transforms(data)
        return X, A, y

    def get_train_data(self):
        X, A, y = self[0]
        return X, A, y, ndl.Tensor(self.idx_train, device=X.device, dtype="int32")

    def get_val_data(self):
        X, A, y = self[0]
        return X, A, y, ndl.Tensor(self.idx_val, device=X.device, dtype="int32")

    def get_test_data(self):
        X, A, y = self[0]
        return X, A, y, ndl.Tensor(self.idx_test, device=X.device, dtype="int32")