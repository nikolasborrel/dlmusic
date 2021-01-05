# Reading material: Christopher Olah's walk-through http://colah.github.io/posts/2015-08-Understanding-LSTMs/
# See also last part of week 5 exercise 5.1-EXE_REcurrent-Neural-Networks.ipynb

# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleLSTMNet(nn.Module):
    def __init__(self, vocab_size):
        super(SimpleLSTMNet, self).__init__()
        
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

class MusicLSTMNet(nn.Module):
    def __init__(self, vocab_size):
        super(MusicLSTMNet, self).__init__()
        
        # Recurrent layer
        self.lstm1 = nn.LSTM(input_size=vocab_size,
                             hidden_size=256,
                             num_layers=3,
                             bidirectional=False)

        self.drop1 = nn.Dropout(0.3)

        self.lstm2 = nn.LSTM(input_size=256,
                             hidden_size=128,
                             num_layers=3,
                             bidirectional=False)
        
        self.drop2 = nn.Dropout(0.3)

        
        # Output layer
        self.l_out = nn.Linear(in_features=128,
                               out_features=vocab_size,
                               bias=False)
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # RNN returns output and last hidden state
        x, (h, c) = self.lstm1(x)        
        #x = self.drop1(x)
        
        x, (h, c) = self.lstm2(x)
        #x = self.drop2(x)

        # Flatten output for feed-forward layer
        x = x.view(-1, self.lstm2.hidden_size)

        # Output layer
        x = self.l_out(x)

        #x = self.softmax(x)
        
        return x

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, \
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), \
                         self.hidden_size) #.to(device)

        c0 = torch.zeros(self.num_layers, x.size(0), \
                         self.hidden_size) #.to(device)

        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -1, :])
        return out