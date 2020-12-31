import numpy as np
import matplotlib.pyplot as plt
from models.model_lstm import MyRecurrentNet
from loaders.dataloader_midi import create_dataset_from_midi
from training.train_lstm import train_lstm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Set seed such that we always get the same dataset
np.random.seed(42)

path_to_midi_dir = '/Users/nikolasborrel/github/midi_data_out/CLEAN_MIDI_BM_small/'

#instruments_to_extract = (54, 34) # voice and bass
instruments_to_extract = (0, 1) # voice and bass

# Hyper-parameters
num_epochs = 200
training_set, validation_set, test_set, tokenizer \
    = create_dataset_from_midi(path_to_midi_dir, instruments_to_extract, print_info=True)
encoder_decoder = tokenizer.encoder_decoder
num_sequences = tokenizer.song_count
vocab_size = tokenizer.vocab_size

#training_set, validation_set, test_set, word_to_idx, idx_to_word, num_sequences, vocab_size = load_dummy_dataset(True)

# Initialize a new LSTM network
net = MyRecurrentNet(vocab_size)
training_loss, validation_loss = train_lstm(net, num_epochs, training_set, validation_set, vocab_size, encoder_decoder)

# Plot training and validation loss
epoch = np.arange(len(training_loss))
plt.figure()
plt.plot(epoch, training_loss, 'r', label='Training loss',)
plt.plot(epoch, validation_loss, 'b', label='Validation loss')
plt.legend()
plt.xlabel('Epoch'), plt.ylabel('NLL')
plt.show()