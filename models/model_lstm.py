# Reading material: Christopher Olah's walk-through http://colah.github.io/posts/2015-08-Understanding-LSTMs/
# See also last part of week 5 exercise 5.1-EXE_REcurrent-Neural-Networks.ipynb

# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f">> Using device: {DEVICE}")


class MusicLSTMNet(nn.Module):
    def __init__(self, vocab_size, hidden_size=250, num_layers=2, dropout_prob=0.3):
        super(MusicLSTMNet, self).__init__()
        self.lstm1 = nn.LSTM(input_size=vocab_size,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             batch_first=True,
                             bidirectional=False,
                             dropout=dropout_prob).to(DEVICE)
        self.init_normal()
        self.drop1 = nn.Dropout(dropout_prob)
        
        # Output layer
        self.l_out = nn.Linear(in_features=hidden_size,
                               out_features=vocab_size,
                               bias=False).to(DEVICE)
        
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # RNN returns output and last hidden state
        x, (h, c) = self.lstm1(x)   
        x = x.to(DEVICE)     
        #x = self.drop1(x)
        
        #x, (h, c) = self.lstm2(x)
        #x = self.drop2(x)

        # Flatten output for feed-forward layer
        x = x.contiguous().view(-1, self.lstm1.hidden_size)

        # Output layer
        x = self.l_out(x) #output is in logits

        #x = self.softmax(x) Not needed CrossEntropy inputs raw logits
        return x
    
    def init_normal(self):
        for layer_p in self.lstm1._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    #print(p, self.lstm1.__getattr__(p))
                    nn.init.normal(self.lstm1.__getattr__(p), 0.0, 0.02)
                    #print(p, self.lstm1.__getattr__(p))


##################################################################################################
############ Test scripts ########################################################################

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=250, num_layers=2):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers,
                            batch_first=False).to(DEVICE)
        self.fc = nn.Linear(in_features=hidden_size, 
                            out_features=input_size).to(DEVICE)
    
    def forward(self, x):
        '''
        h0 = torch.zeros(self.num_layers, x.size(0), \
                         self.hidden_size).to(DEVICE)

        c0 = torch.zeros(self.num_layers, x.size(0), \
                         self.hidden_size).to(DEVICE)
        '''
        out, (h, c) = self.lstm(x)
        out.to(DEVICE)
        out = out.view(-1, self.lstm.hidden_size)
        out = self.fc(out)
        #out = self.fc(out[:, -1, :])
        return out

class SimpleLSTMNet(nn.Module):
    def __init__(self, vocab_size, hidden_size=512):
        super(SimpleLSTMNet, self).__init__()
        
        # Recurrent layer
        # YOUR CODE HERE!
        self.lstm = nn.LSTM(input_size=vocab_size,
                            hidden_size=hidden_size,
                            num_layers=1,
                            bidirectional=False)
        
        # Output layer
        self.l_out = nn.Linear(in_features=hidden_size,
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

class MusicLSTMNet2(nn.Module):
    def __init__(self, vocab_size):
        super(MusicLSTMNet2, self).__init__()
        hs = [100, 50]
        # Recurrent layer
        self.lstm1 = nn.LSTM(input_size=vocab_size,
                             hidden_size=hs[0],
                             num_layers=3,
                             bidirectional=False).to(DEVICE)

        self.drop1 = nn.Dropout(0.3)

        self.lstm2 = nn.LSTM(input_size=hs[0],
                             hidden_size=hs[1],
                             num_layers=3,
                             bidirectional=False).to(DEVICE)
        
        self.drop2 = nn.Dropout(0.3)

        
        # Output layer
        self.l_out = nn.Linear(in_features=hs[1],
                               out_features=vocab_size,
                               bias=False)
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # RNN returns output and last hidden state
        x, (h, c) = self.lstm1(x)   
        x.to(DEVICE)     
        x = self.drop1(x)
        
        x, (h, c) = self.lstm2(x)
        x = self.drop2(x)

        # Flatten output for feed-forward layer
        x = x.view(-1, self.lstm2.hidden_size)

        # Output layer
        x = self.l_out(x)

        #x = self.softmax(x)
        return x
