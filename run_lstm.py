import sys
sys.path.append('../note_seq') # needed unless installing forked lib from github

import numpy as np
import matplotlib.pyplot as plt
from models.model_lstm import DEVICE, MusicLSTMNet
from loaders.dataloader_midi import create_dataset_from_midi, create_dataset_from_midi_one_song
from training.train_lstm import train_lstm, train_lstm_scheduled_lr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils.paths as paths
from utils.tools import write_json
from models.distributions import get_histograms_from_dataloader

#model_name = '3_layer_256_20_epochs_lr_0.7e-4_batch_16_dropout_0.4_chunks20'
model_name = 'Test'
dataset_path = paths.midi_dir

# Set seed such that we always get the same dataset
np.random.seed(42)

# Encoding-Tokenizing parameters
instruments = [0,1]
lead_instrument   = ('melody',instruments[0])
accomp_instrument = ('bass',instruments[1])

tokenizer_kwargs = {'split_in_bar_chunks': 4, 
                    'steps_per_quarter': 1, 
                    'min_note': 60, #60
                    'max_note':72} #72

length_per_seq =  tokenizer_kwargs['steps_per_quarter'] * 4 * tokenizer_kwargs['split_in_bar_chunks']
print('length_per_seq: ', length_per_seq)

# Dataset creation, splitting and batching parameters
dataset_split_kwargs = {'p_train': 0.7, 'p_val': 0.3, 'p_test': 0.0,
                        'batch_size': 16,
                        'eval_batch_size': 16}

train, val, test, t = create_dataset_from_midi(dataset_path, 
                                               lead_instrument, 
                                               accomp_instrument, 
                                               print_info=True,
                                               **tokenizer_kwargs, 
                                               **dataset_split_kwargs)

encoder_decoder = t.encoder_decoder
num_sequences = t.song_count

#LSTM hyperparameters
vocab_size = t.vocab_size
num_epochs = 20
learning_rate = 0.7e-4 
dropout_prob=0.5
num_layers=2
hidden_size=256

print('train Histogram')
get_histograms_from_dataloader(train, vocab_size=vocab_size, plot=True)
print('Validation Histogram')
get_histograms_from_dataloader(val, vocab_size=vocab_size, plot=True)
#net = MusicLSTMNet2(vocab_size).to(DEVICE) # Initialize LSTM network
#net = LSTM(vocab_size,hidden_size=500, num_layers=5)
net = MusicLSTMNet(vocab_size,
                   hidden_size=hidden_size, 
                   num_layers=num_layers, 
                   dropout_prob=dropout_prob).to(DEVICE)
print(net)

training_loss, validation_loss = train_lstm(net, num_epochs, train, val, 
                                            vocab_size, encoder_decoder,
                                            lr=learning_rate)

# save net
torch.save(net.cpu(), paths.model_serialized_dir + model_name + '.pt')

# Create params dictionary to be saved with all hyperparameters
params = dict(vocab_size=vocab_size,
              num_epochs=num_epochs,
              lr=learning_rate,
              num_sequences=num_sequences,
              length_per_seq=length_per_seq,
              dropout_prob=dropout_prob,
              num_layers=num_layers,
              hidden_size=hidden_size)

params.update(tokenizer_kwargs)
params.update(dataset_split_kwargs)
params['training_loss'] = training_loss
params['validation_loss'] = validation_loss
write_json(params, paths.model_serialized_dir + model_name + '.json')

# Plot training and validation loss
epoch = np.arange(len(training_loss))
plt.figure()
plt.plot(epoch, training_loss, 'r', label='Training loss',)
plt.plot(epoch, validation_loss, 'b', label='Validation loss')
plt.legend()
plt.xlabel('Epoch'), plt.ylabel('NLL')
plt.show()

print('training loss converged to:   ', training_loss[-1])
print('validation loss converged to: ', validation_loss[-1])
#%%
# Compare target y with output from forward pass from first sample of test_set
# TO DO: Implement a SCORE for all test data
count=0
for inputs_one_hot, targets_idx in train:
    if count == 0:
        inputs_one_hot = inputs_one_hot.to(DEVICE)
        break
    count +=1

print(inputs_one_hot.device)
outputs = net.to(DEVICE).forward(inputs_one_hot[0].unsqueeze(0))
x = train.dataset.inputs[0]._events
y = train.dataset.targets[0]._events

events = []
for output in outputs.cpu().detach().numpy():
    label = np.argmax(output)
    events.append(t._encoder_decoder.class_index_to_event(label, events))

print('target:    ', y[1:])
print('predicted: ', events)
# %%
