import sys
sys.path.append('../note_seq') # needed unless installing forked lib from github

import numpy as np
import matplotlib.pyplot as plt
from models.model_lstm import MusicLSTMNet
from loaders.dataloader_midi import create_dataset_from_midi
from training.train_lstm import train_lstm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils.paths as paths

# Set seed such that we always get the same dataset
np.random.seed(42)

instruments = [0,1]
lead_instrument   = ('melody',instruments[0])
accomp_instrument = ('bass',instruments[1])

split_in_bar_chunks = 8

# Hyper-parameters
num_epochs = 100
train, val, test, t = create_dataset_from_midi(paths.midi_dir_small, 
                                               lead_instrument, 
                                               accomp_instrument, 
                                               split_in_bar_chunks, 
                                               print_info=True)
encoder_decoder = t.encoder_decoder
num_sequences = t.song_count


# Hyperparameters

num_epochs = 100
batch_size = 16

vocab_size = t.vocab_size

#%%

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#Train LSTM imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

        print('x: ',x.shape)
        print('h: ', h.shape)
        print('c: ', c.shape)
        
        # Flatten output for feed-forward layer
        x = x.view(-1, self.lstm.hidden_size)
        
        # Output layer
        x = self.l_out(x)
        
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
                         self.hidden_size).to(device)

        c0 = torch.zeros(self.num_layers, x.size(0), \
                         self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -1, :])
        return out
#%%
# Initialize a new LSTM network
net = SimpleLSTMNet(vocab_size)
#def train_lstm(net, num_epochs, training_set, validation_set, vocab_size, encoder_decoder):
    # Define a loss function and optimizer for this problem
    # YOUR CODE HERE!
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Track loss
training_loss, validation_loss = [], []

# For each epoch
for i in range(num_epochs):
    
    # Track loss
    epoch_training_loss = 0
    epoch_validation_loss = 0
    
    net.eval()
        
    # For each sentence in validation set
    for inputs_one_hot, targets_idx in validation_set:
        optimizer.zero_grad()

        # DIMENSION inputs_one_hot
        # (batch_size, seq_len, input_size)

        # DIMENSION targets_idx
        # (batch_size, seq_len)

        
        # # One-hot encode input and target sequence
        # inputs_one_hot, _ = encoder_decoder.get_inputs_batch(input_and_target[0])  # MELODY
        # _, targets_idx = encoder_decoder.encode(input_and_target[1])               # BASS
        
        # # NBJ comment: 3 dimensional array needed with dimension (batch_size, seq_len, input_size)
        # # TODO: batch size of one currently!
        # inputs_one_hot_3D = [inputs_one_hot]

        # # Convert input to tensor            
        # inputs_one_hot_3D = torch.Tensor(inputs_one_hot_3D) # permute not needed
        
        # # Convert target to tensor
        # targets_idx = torch.LongTensor(targets_idx)
        
        # Forward pass
        # YOUR CODE HERE!
        outputs = net.forward(inputs_one_hot)
        
        # Compute loss
        # YOUR CODE HERE!

        loss = criterion(outputs, targets_idx)
        
        # Update loss
        epoch_validation_loss += loss.detach().numpy()
    
    net.train()
    
    # For each sentence in training set
    for inputs_one_hot, targets_idx in training_set:
        optimizer.zero_grad()

        # DIMENSION inputs_one_hot
        # (batch_size, seq_len, input_size)

        # DIMENSION targets_idx
        # (batch_size, seq_len)


        # # One-hot encode input and target sequence
        # inputs_one_hot, _ = encoder_decoder.get_inputs_batch(input_and_target[0])
        # _, targets_idx = encoder_decoder.encode(input_and_target[1])
        
        # # NBJ comment: 3 dimensional array needed with dimension (batch_size, seq_len, input_size)
        # # TODO: batch size of one currently!
        # inputs_one_hot_3D = [inputs_one_hot]

        # # Convert input to tensor
        # inputs_one_hot_3D = torch.Tensor(inputs_one_hot_3D) # permute not needed
        
        # # Convert target to tensor
        # targets_idx = torch.LongTensor(targets_idx)
        
        # Forward pass
        # YOUR CODE HERE!
        outputs = net.forward(inputs_one_hot)
        
        # Compute loss
        # YOUR CODE HERE!
        loss = criterion(outputs, targets_idx)
        
        # Backward pass
        # YOUR CODE HERE!
        # zero grad, backward, step...
        loss.backward()
        optimizer.step()
        
        # Update loss
        epoch_training_loss += loss.detach().numpy()
        
    # Save loss for plot
    training_loss.append(epoch_training_loss/len(training_set))
    validation_loss.append(epoch_validation_loss/len(validation_set))

    # Print loss every 10 epochs
    if i % 10 == 0:
        print(f'Epoch {i}, training loss: {training_loss[-1]}, validation loss: {validation_loss[-1]}')

    #return training_loss, validation_loss