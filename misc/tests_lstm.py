import numpy as np
import matplotlib.pyplot as plt

from models.model_lstm import *
from loaders.dataloader_dummy import load_dummy_dataset
from utils.hot_encoder import one_hot_encode_sequence, one_hot_encode
from training.train_lstm import train_lstm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

training_set, validation_set, test_set, word_to_idx, idx_to_word, num_sequences, vocab_size = load_dummy_dataset(True)

## Test one_hot
test_word = one_hot_encode(word_to_idx['a'], vocab_size)
print(f'Our one-hot encoding of \'a\' has shape {test_word.shape}.')

test_sentence = one_hot_encode_sequence(['a', 'b'], vocab_size, word_to_idx)
print(f'Our one-hot encoding of \'a b\' has shape {test_sentence.shape}.')

## VISUALIZE
net = MyRecurrentNet(vocab_size)

# Get first sentence in test set
inputs, targets = test_set[1]

# One-hot encode input and target sequence
inputs_one_hot = one_hot_encode_sequence(inputs, vocab_size, word_to_idx)

targets_idx = [word_to_idx[word] for word in targets]

# Convert input to tensor
inputs_one_hot = torch.Tensor(inputs_one_hot)
inputs_one_hot = inputs_one_hot.permute(0, 2, 1)

# Convert target to tensor
targets_idx = torch.LongTensor(targets_idx)

# Forward pass
outputs = net.forward(inputs_one_hot).data.numpy()

print('\nInput sequence:')
print(inputs)

print('\nTarget sequence:')
print(targets)

print('\nPredicted sequence:')
print([idx_to_word[np.argmax(output)] for output in outputs])