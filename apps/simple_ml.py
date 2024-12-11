import numpy as np
import sys
sys.path.append("python/")
import needle as ndl

def train_gcn(model, dataloader, num_epochs, learning_rate):
    """
    Train the GCN model using the provided DataLoader.

    Args:
        model: The GCN model instance.
        dataloader: DataLoader instance providing batches of data.
        num_epochs: Number of training epochs.
        learning_rate: Learning rate for gradient descent.

    Returns:
        model: Trained GCN model.
    """
    for epoch in range(num_epochs):
        epoch_loss = 0
        correct_predictions = 0
        total_samples = 0

        for X_batch, y_batch, adjacency_batch in dataloader:
            # One-hot encode labels
            y_indices = y_batch.data.numpy().astype(int)  # Convert Tensor to NumPy array
            num_classes = len(np.unique(dataloader.y.data))
            y_one_hot = np.eye(num_classes)[y_indices]
            y_tensor = ndl.Tensor(y_one_hot, device=y_batch.device)

            # Forward pass
            logits = model(X_batch, adjacency_batch)

            # Compute loss
            loss = softmax_loss(logits, y_tensor)

            # Backward pass
            loss.backward()

            # Update weights
            for param in model.parameters():
                param.data -= learning_rate * param.grad.data
                param.grad = None  # Reset gradients

            # Accumulate loss
            epoch_loss += loss.data

            # Compute training accuracy
            predictions = np.argmax(logits.data.numpy(), axis=1)
            correct_predictions += np.sum(predictions == y_indices)
            total_samples += y_batch.shape[0]

        # Calculate average loss and accuracy for the epoch
        avg_loss = epoch_loss / X_batch.shape[0]
        train_accuracy = correct_predictions / total_samples
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}, Train Accuracy: {train_accuracy * 100:.2f}%")

    return model

def evaluate_gcn(model, dataloader):
    """
    Evaluate the GCN model using the provided DataLoader.
    """
    correct_predictions = 0
    total_samples = 0

    for X_batch, y_batch, adjacency_batch in dataloader:
        # One-hot encode labels
        y_indices = y_batch.data.numpy().astype(int)  # Convert Tensor to NumPy array
        num_classes = len(np.unique(dataloader.y.data))
        y_one_hot = np.eye(num_classes)[y_indices]
        y_tensor = ndl.Tensor(y_one_hot, device=y_batch.device)

        # Forward pass
        logits = model(X_batch, adjacency_batch)

        # Compute training accuracy
        predictions = np.argmax(logits.data.numpy(), axis=1)
        correct_predictions += np.sum(predictions == y_indices)
        total_samples += y_batch.shape[0]

    accuracy = correct_predictions / total_samples
    return accuracy

def split_data(X, y, adjacency_matrix, train_ratio=0.8):
    num_samples = X.shape[0]
    num_train = int(train_ratio * num_samples)
    
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    train_indices = indices[:num_train]
    test_indices = indices[num_train:]

    X_train = X[train_indices]
    y_train = y[train_indices]
    adj_train = adjacency_matrix[train_indices][:, train_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]
    adj_test = adjacency_matrix[test_indices][:, test_indices]

    return X_train, y_train, adj_train, X_test, y_test, adj_test

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
    m = Z.shape[0]
    exps = ndl.ops.exp(Z)
    log = ndl.ops.log(ndl.ops.summation(exps, axes=(1, )))
    Z1 = ndl.ops.summation(log)
    Z2 = ndl.ops.summation(Z * y_one_hot)
    return (Z1 - Z2) / m