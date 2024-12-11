import numpy as np
from ..autograd import Tensor

class DataLoader:
    """
    A simple data loader for batching and shuffling CoraDataset data.

    Args:
        X (np.array): Feature matrix.
        y (np.array): Labels.
        adjacency_matrix (np.array): Dense adjacency matrix.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data at the beginning of each iteration.
    """

    def __init__(self, X, y, adjacency_matrix, batch_size=1, shuffle=False, device = None):
        self.X = X
        self.y = y
        self.adjacency_matrix = adjacency_matrix
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = X.shape[0]
        self.order = np.arange(self.num_samples)
        self.device = device

    def __iter__(self):
        """Initializes the iterator."""
        if self.shuffle:
            np.random.shuffle(self.order)
        self.index = 0
        return self

    def __next__(self):
        """Fetches the next batch of data."""
        if self.index >= self.num_samples:
            raise StopIteration

        # Get indices for the current batch
        start = self.index
        end = min(self.index + self.batch_size, self.num_samples)
        batch_indices = self.order[start:end]

        # Fetch the batch
        X_batch = self.X[batch_indices]
        y_batch = self.y[batch_indices]
        adjacency_batch = self.adjacency_matrix[batch_indices][:, batch_indices]

        # Convert to Tensors
        X_batch = Tensor(X_batch, device = self.device)
        y_batch = Tensor(y_batch, device = self.device)
        adjacency_batch = Tensor(adjacency_batch, device = self.device)

        self.index += self.batch_size
        return X_batch, y_batch, adjacency_batch