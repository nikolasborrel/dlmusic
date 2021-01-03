# Reading material: Christopher Olah's walk-through http://colah.github.io/posts/2015-08-Understanding-LSTMs/
# See also last part of week 5 exercise 5.1-EXE_REcurrent-Neural-Networks.ipynb

# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MyRecurrentNet(nn.Module):
    def __init__(self, vocab_size):
        super(MyRecurrentNet, self).__init__()
        
        # Recurrent layer
        # YOUR CODE HERE!
        self.lstm = nn.LSTM(input_size=vocab_size,
                            hidden_size=50,
                            num_layers=1,
                            bidirectional=False)
        
        # Output layer
        self.l_out = nn.Linear(in_features=50,
                               out_features=vocab_size,
                               bias=False)
        
    def forward(self, x):
        # RNN returns output and last hidden state
        x, (h, c) = self.lstm(x)
        
        # Flatten output for feed-forward layer
        x = x.view(-1, self.lstm.hidden_size)
        
        # Output layer
        x = self.l_out(x)
        
        return x