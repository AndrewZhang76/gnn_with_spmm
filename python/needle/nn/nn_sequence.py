"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return (1 + ops.exp(-x)) ** (-1)
        ### END YOUR SOLUTION

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.device = device
        self.dtype = dtype
        self.bias = bias
        self.hidden_size = hidden_size
        
        bound = np.sqrt(1 / hidden_size)
        
        
        self.W_ih = Parameter(init.rand(input_size, hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
        
        
        self.bias_ih = None
        self.bias_hh = None
        if bias:
          self.bias_ih = Parameter(init.rand(hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
          self.bias_hh = Parameter(init.rand(hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
        
        
        if nonlinearity == "tanh":
          self.nonlinearity = ops.Tanh()
        elif nonlinearity == "relu":
          self.nonlinearity = ops.ReLU()

        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
          h = init.zeros(X.shape[0], self.hidden_size, device=self.device, dtype=self.dtype)
        out = X @ self.W_ih + h @ self.W_hh
        if self.bias:
          out = out + self.bias_ih.reshape((1, self.hidden_size)).broadcast_to((X.shape[0], self.hidden_size)) + self.bias_hh.reshape((1, self.hidden_size)).broadcast_to((X.shape[0], self.hidden_size))
        return self.nonlinearity(out)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.hidden_size = hidden_size
        self.num_layers = num_layers 
        self.device = device
        self.dtype = dtype
        self.rnn_cells = [RNNCell(input_size, hidden_size, bias=bias, nonlinearity=nonlinearity, device=device, dtype=dtype)]
        for _ in range(num_layers-1):
          self.rnn_cells.append(RNNCell(hidden_size, hidden_size, bias=bias, nonlinearity=nonlinearity, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h0 is None:
          h0 = []
          for _ in range(self.num_layers):
            h0.append(init.zeros(X.shape[1], self.hidden_size, device=self.device, dtype=self.dtype))
        else:
          h0 = tuple(ops.split(h0, 0))
        h_n = []
        X_list = list(tuple(ops.split(X, 0)))
        for i in range(self.num_layers):
          h = h0[i]
          cell = self.rnn_cells[i]
          for idx, input in enumerate(X_list):
            h = cell(input, h)
            X_list[idx] = h
          h_n.append(h)
        output = ops.stack(X_list, 0)
        hidden_state = ops.stack(h_n, 0)
        return output, hidden_state 
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.device = device
        self.dtype = dtype
        self.bias = bias
        self.hidden_size = hidden_size
        
        bound = np.sqrt(1 / hidden_size)
        
        
        self.W_ih = Parameter(init.rand(input_size, 4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
        self.W_hh = Parameter(init.rand(hidden_size, 4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
        
        
        self.bias_ih = None
        self.bias_hh = None
        if bias:
          self.bias_ih = Parameter(init.rand(4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
          self.bias_hh = Parameter(init.rand(4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
        
        self.sigmoid = Sigmoid()
        self.Tanh = ops.Tanh()

        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
          h0 = init.zeros(X.shape[0], self.hidden_size, device=self.device, dtype=self.dtype)
          c0 = init.zeros(X.shape[0], self.hidden_size, device=self.device, dtype=self.dtype)
        else:
          h0,c0 = h
        ifgh = X @ self.W_ih + h0 @ self.W_hh
        if self.bias:
          ifgh += self.bias_ih.reshape((1, 4 * self.hidden_size)).broadcast_to((X.shape[0], 4 * self.hidden_size))
          ifgh += self.bias_hh.reshape((1, 4 * self.hidden_size)).broadcast_to((X.shape[0], 4 * self.hidden_size))
        ifgh = ops.split(ifgh, 1)
        i = tuple([ifgh[i] for i in range(0, self.hidden_size)])
        f = tuple([ifgh[i] for i in range(self.hidden_size, 2*self.hidden_size)])
        g = tuple([ifgh[i] for i in range(2*self.hidden_size, 3*self.hidden_size)])
        o = tuple([ifgh[i] for i in range(3*self.hidden_size, 4*self.hidden_size)])
        i,f,g,o = self.sigmoid(ops.stack(i, 1)), self.sigmoid(ops.stack(f, 1)), self.Tanh(ops.stack(g, 1)), self.sigmoid(ops.stack(o, 1))
        c = f * c0 + i * g
        h = o * self.Tanh(c)
        return (h, c)

        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_cells = [LSTMCell(input_size, hidden_size, bias, device, dtype)]
        for i in range(num_layers - 1):
            self.lstm_cells.append(LSTMCell(hidden_size, hidden_size, bias, device, dtype))
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
          h0 = init.zeros(self.num_layers, X.shape[1], self.hidden_size, device=X.device, dtype=X.dtype)
          c0 = init.zeros(self.num_layers, X.shape[1], self.hidden_size, device=X.device, dtype=X.dtype)
        else:
          h0, c0 = h
        ht, ct = ops.split(h0, 0), ops.split(c0, 0)
        hidden_states  = []
        X_tuple = ops.split(X, 0)
        for t in range(X.shape[0]):
          xt = X_tuple[t]
            
          # Iterate over LSTM layers
          for layer in range(self.num_layers):
              # Get LSTM cell for current layer
              lstm_cell = self.lstm_cells[layer]
              
              # Update hidden and cell states for current layer
              ht_next, ct_next = lstm_cell(xt, (ht[layer], ct[layer]))
              
              # Update hidden and cell states for next layer
              if layer == 0:
                  ht_layer = [ht_next]
                  ct_layer = [ct_next]
              else:
                  ht_layer.append(ht_next)
                  ct_layer.append(ct_next)
              xt = ht_next # Output of current layer becomes input for next layer

          # Stack the updated hidden states for all layers
          ht = ht_layer
          ct = ct_layer
          # Append the hidden state of the last layer for this time step
          hidden_states.append(ht[-1])
      
        # Stack hidden states to get the output
        output = ops.stack(hidden_states, axis=0)
        h_n = ops.stack(ht, axis=0)
        c_n = ops.stack(ct, axis=0)

        return output, (h_n, c_n)

        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.randn(num_embeddings, embedding_dim, mean=0.0, std=1.0, device=device, dtype=dtype, requires_grad=True))
        self.device = device
        self.dtype = dtype
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs = x.shape
        x_one_hot = init.one_hot(self.weight.shape[0], x, device=self.device, dtype=self.dtype).reshape((seq_len * bs, self.weight.shape[0]))
        return (x_one_hot @ self.weight).reshape((seq_len, bs, self.weight.shape[1]))
        ### END YOUR SOLUTION