import numpy as np

class Transform:
    def __call__(self, x):
        raise NotImplementedError

##################### For Cora Dataset #####################
class ToTensor(Transform):
    def __init__(self, device=None, dtype="float32"):
        self.device = device
        self.dtype = dtype

    def __call__(self, data):
        # tuple (X, y, A) 
        X, y, A = data
        import needle as ndl
        X = ndl.Tensor(X, device=self.device, dtype=self.dtype)
        y = ndl.Tensor(y, device=self.device, dtype="int32")        # Labels are integers
        A = ndl.Tensor(A, device=self.device, dtype=self.dtype)
        return X, y, A
    
class Compose(Transform):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data

##################### For image Dataset #####################
class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as an H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
          return np.flip(img, axis=1)
        else:
          return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        H, W = img.shape[0], img.shape[1]
        padded_img = np.zeros((H+2*self.padding, W+2*self.padding, img.shape[2]))
        padded_img[self.padding:self.padding+H, self.padding:self.padding+W, :] = img
        cropped_img = padded_img[self.padding+shift_x:self.padding+shift_x+H, self.padding+shift_y:self.padding+shift_y+W, :]
        return cropped_img
        ### END YOUR SOLUTION