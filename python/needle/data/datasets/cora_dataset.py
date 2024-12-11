import numpy as np
import needle as ndl

class CoraDataset:
    def __init__(self, feature_path, label_path, adj_path, device=ndl.cpu()):
        self.X = np.load(feature_path)      # [N, F]
        self.y = np.load(label_path)        # [N]
        A_dense = np.load(adj_path)         # [N, N]

        # Convert to needle Tensors
        self.X = ndl.Tensor(self.X, device=device, dtype="float32")
        self.y = ndl.Tensor(self.y, device=device, dtype="int32")
        self.A = ndl.Tensor(A_dense, device=device, dtype="float32")

        N = self.X.shape[0]
        # Example splits:
        self.idx_train = ndl.Tensor(np.arange(0, 140), device=device, dtype="int32")
        self.idx_val = ndl.Tensor(np.arange(140, 640), device=device, dtype="int32")
        self.idx_test = ndl.Tensor(np.arange(1708, 2708), device=device, dtype="int32")

    def get_train_data(self):
        return self.X, self.A, self.y, self.idx_train

    def get_val_data(self):
        return self.X, self.A, self.y, self.idx_val

    def get_test_data(self):
        return self.X, self.A, self.y, self.idx_test