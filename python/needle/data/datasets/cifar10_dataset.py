import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        self.base_folder = base_folder
        self.train = train
        self.transforms = transforms

        if self.train:
            data_files = [f"data_batch_{i}" for i in range(1, 6)]
        else:
            data_files = ["test_batch"]

        self.X = []
        self.y = []
        for file_name in data_files:
            file_path = os.path.join(self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                data_dict = pickle.load(f, encoding='bytes')
                self.X.extend(data_dict[b'data'])
                self.y.extend(data_dict[b'labels'])

        self.X = np.array(self.X, dtype=np.float32).reshape(-1, 3, 32, 32) / 255.0 
        self.y = np.array(self.y, dtype=np.int64)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        X_item = self.X[index]
        y_item = self.y[index]

        if self.transforms:
            X_item = self.transforms(X_item)

        return X_item, y_item
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.X)
        ### END YOUR SOLUTION
