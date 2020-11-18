import numpy as np
import matplotlib.pyplot as plt

from models.model_lstm import MyRecurrentNet
from loaders.dataloader_dummy import load_dummy_dataset
from utils.hot_encoder import one_hot_encode_sequence
from training.train_lstm import train_lstm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Set seed such that we always get the same dataset
np.random.seed(42)

# Hyper-parameters
num_epochs = 200
training_set, validation_set, test_set, word_to_idx, idx_to_word, num_sequences, vocab_size = load_dummy_dataset(True)

# Initialize a new LSTM network
net = MyRecurrentNet(vocab_size)
training_loss, validation_loss = train_lstm(net, num_epochs, training_set, validation_set, vocab_size, word_to_idx)

# Plot training and validation loss
epoch = np.arange(len(training_loss))
plt.figure()
plt.plot(epoch, training_loss, 'r', label='Training loss',)
plt.plot(epoch, validation_loss, 'b', label='Validation loss')
plt.legend()
plt.xlabel('Epoch'), plt.ylabel('NLL')
plt.show()