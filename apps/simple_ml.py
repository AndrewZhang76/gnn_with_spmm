"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl
import needle.nn as nn
from apps.models import *
import time
device = ndl.cpu()

### Cora training ###

def train_cora(model, dataset, n_epochs=200, optimizer=ndl.optim.Adam, lr=0.01, weight_decay=5e-4):
    X, A, y, idx_train = dataset.get_train_data()
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SoftmaxLoss()

    for epoch in range(n_epochs):
        model.train()
        opt.zero_grad()

        out = model(X, A)
        train_out = out[idx_train, :]
        train_labels = y[idx_train]
        loss = loss_fn(train_out, train_labels)

        loss.backward()
        opt.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss.numpy():.4f}")

def evaluate_cora(model, dataset, split='test'):
    model.eval()
    if split == 'train':
        X, A, y, idx = dataset.get_train_data()
    elif split == 'val':
        X, A, y, idx = dataset.get_val_data()
    else:
        X, A, y, idx = dataset.get_test_data()

    out = model(X, A)
    out_split = out[idx, :]
    labels_split = y[idx].numpy()
    preds = out_split.numpy().argmax(axis=1)
    acc = (preds == labels_split).mean()
    print(f"{split.capitalize()} Accuracy: {acc:.4f}")
    return acc

def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    DATA_TYPES = {
        0x08: "B",
        0x09: "b",
        0x0B: "h",
        0x0C: "i",
        0x0D: "f",
        0x0E: "d",
    }
    with gzip.open(label_filename, 'rb') as file:
      zeros, dtype, n_dimensions = struct.unpack(">HBB", file.read(4))
      file.seek(4 * n_dimensions, 1)
      labels = array.array(DATA_TYPES[dtype], file.read())
      labels = np.array(labels, np.uint8)
    with gzip.open(image_filename, 'rb') as file:
      magic, dtype, n = struct.unpack(">HBB", file.read(4))
      dimension_sizes = struct.unpack(
        f"> {'I' * n}", file.read(4 * n)
      )
      data = array.array(DATA_TYPES[dtype], file.read())
      image_data = np.array(data, dtype=np.float32).reshape(-1, mul(*dimension_sizes[1:])) / 255
    return (image_data, labels)
    ### END YOUR CODE

def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    m = Z.shape[0]
    exps = ndl.ops.exp(Z)
    log = ndl.ops.log(ndl.ops.summation(exps, axes=(1, )))
    Z1 = ndl.ops.summation(log)
    Z2 = ndl.ops.summation(Z * y_one_hot)
    return (Z1 - Z2) / m
    ### END YOUR SOLUTION

def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    m = X.shape[0]
    for i in range(0, m, batch):
        X_batch = X[i : i+batch]
        y_batch = y[i : i+batch]
        X_batch = ndl.Tensor(X_batch)
        Z1 = ndl.ops.relu(X_batch @ W1)
        Z = Z1 @ W2
        y_one_hot = np.zeros(Z.shape, dtype="float32")
        y_one_hot[np.arange(Z.shape[0]),y_batch] = 1
        loss = softmax_loss(Z, ndl.Tensor(y_one_hot))
        loss.backward()
        w1_grad = W1.grad
        w2_grad = W2.grad
        W1 = np.array(W1) - lr * w1_grad
        W2 = np.array(W2) - lr * w2_grad
        W1 = ndl.Tensor(W1)
        W2 = ndl.Tensor(W2)
    return W1, W2
    ### END YOUR SOLUTION

### CIFAR-10 training ###
def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    total_acc, total_loss = 0, 0
    print("check2")
    if opt is not None:
      model.train()
      for batch in dataloader:
          opt.reset_grad()
          X, y = batch
          X, y = ndl.Tensor(X, device=device), ndl.Tensor(y, device=device)
          out = model(X)
          loss = loss_fn()(out, y)
          loss.backward()
          opt.step()
          total_acc += np.sum(np.argmax(out.numpy(), axis=1) == y.numpy())
          total_loss += loss.data.numpy() * y.shape[0]
    else:
      model.eval()
      for batch in dataloader:
          X, y = batch
          X, y = ndl.Tensor(X, device=device), ndl.Tensor(y, device=device)
          out = model(X)
          loss = loss_fn()(out, y)
          total_acc += np.sum(np.argmax(out.numpy(), axis=1) == y.numpy())
          total_loss += loss.data.numpy() * y.shape[0]
    return total_acc / len(dataloader.dataset), total_loss / len(dataloader.dataset)
    ### END YOUR SOLUTION


def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
          lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    print('hihihi')
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    print("Done Opt")
    for epoch in range(n_epochs):
        avg_acc, avg_loss = epoch_general_cifar10(dataloader, model, loss_fn=loss_fn, opt=opt)
        print(f"Epoch: {epoch}, Acc: {avg_acc}, Loss: {avg_loss}")
    ### END YOUR SOLUTION


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    avg_acc, avg_loss = epoch_general_cifar10(dataloader, model, loss_fn=loss_fn)
    ### END YOUR SOLUTION


### PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
        clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    losses = []
    accs = []
    dataset_size = 0
    nbatch, batch_size = data.shape
    
    hidden = None
    if opt is not None:
      model.train()
      for i in range(0, nbatch - 1, seq_len):
          x, y = ndl.data.get_batch(data, i, seq_len, device=device, dtype=dtype)
          batch_size = y.shape[0]
          dataset_size += batch_size
          y_pred, hidden = model(x, hidden)
          if isinstance(hidden, tuple):
              h, c = hidden
              hidden = (h.detach(), c.detach())
          else:
              hidden = hidden.detach()
          loss = loss_fn()(y_pred, y)
          opt.reset_grad()
          loss.backward()
          opt.step()
          losses.append(loss.numpy() * batch_size)
          acc = np.sum(y_pred.numpy().argmax(axis = 1) == y.numpy())
          accs.append(acc)
    else:
      model.eval()
      for i in range(0, nbatch - 1, seq_len):
          x, y = ndl.data.get_batch(data, i, seq_len, device=device, dtype=dtype)
          batch_size = y.shape[0]
          dataset_size += batch_size
          y_pred, hidden = model(x, hidden)
          if isinstance(hidden, tuple):
              h, c = hidden
              hidden = (h.detach(), c.detach())
          else:
              hidden = hidden.detach()
          loss = loss_fn()(y_pred, y)
          losses.append(loss.numpy() * batch_size)
          acc = np.sum(y_pred.numpy().argmax(axis = 1) == y.numpy())
          accs.append(acc)
    return np.sum(np.array(accs)) / dataset_size, np.sum(np.array(losses)) / dataset_size
    ### END YOUR SOLUTION


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
          lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
          device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(), lr = lr, weight_decay = weight_decay)

    for i in range(n_epochs):
        avg_acc, avg_loss = epoch_general_ptb(data, model, seq_len=seq_len, loss_fn=loss_fn, opt=opt, device=device, dtype=dtype)
        print("Epoch {} acc is {}, loss is {}.".format(i, avg_acc, avg_loss))
    return avg_acc, avg_loss
    ### END YOUR SOLUTION

def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
        device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    avg_acc, avg_loss = epoch_general_ptb(data, model, seq_len=seq_len, loss_fn=loss_fn, opt=None, device=device, dtype=dtype)
    return avg_acc, avg_loss
    ### END YOUR SOLUTION

### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
