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

max_bars = 16

# Hyper-parameters
num_epochs = 100
training_set, validation_set, test_set, tokenizer \
    = create_dataset_from_midi(paths.midi_dir, lead_instrument, accomp_instrument, max_bars, print_info=True)
encoder_decoder = tokenizer.encoder_decoder
num_sequences = tokenizer.song_count
vocab_size = tokenizer.vocab_size

# Initialize a new LSTM network
net = MusicLSTMNet(vocab_size)
training_loss, validation_loss = train_lstm(net, num_epochs, training_set, validation_set, vocab_size, encoder_decoder)

torch.save(net, paths.model_serialized_dir + 'music_lstm.pt')

# Plot training and validation loss
epoch = np.arange(len(training_loss))
plt.figure()
plt.plot(epoch, training_loss, 'r', label='Training loss',)
plt.plot(epoch, validation_loss, 'b', label='Validation loss')
plt.legend()
plt.xlabel('Epoch'), plt.ylabel('NLL')
plt.show()