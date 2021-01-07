import sys
sys.path.append('../note_seq') # needed unless installing forked lib from github

import numpy as np
import matplotlib.pyplot as plt
from models.model_lstm import MusicLSTMNet, LSTM, SimpleLSTMNet
from loaders.dataloader_midi import create_dataset_from_midi
from training.train_lstm import train_lstm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils.paths as paths
from utils.tools import write_json
from models.distributions import get_histograms_from_dataloader


model_name = 'music_lstm'
dataset_path = paths.midi_dir_small

# Set seed such that we always get the same dataset
np.random.seed(42)

# Encoding-Tokenizing parameters
instruments = [0,1]
lead_instrument   = ('melody',instruments[0])
accomp_instrument = ('bass',instruments[1])

tokenizer_kwargs = {'split_in_bar_chunks': 6, 
                    'steps_per_quarter': 1, 
                    'min_note': 60, 
                    'max_note':72}

length_per_seq =  tokenizer_kwargs['steps_per_quarter'] * 4 * tokenizer_kwargs['split_in_bar_chunks']
print('length_per_seq: ', length_per_seq)

# Dataset creation, splitting and batching parameters
dataset_split_kwargs = {'p_train': 0.7, 'p_val': 0.2, 'p_test': 0.1,
                        'batch_size': 1,
                        'eval_batch_size': 1}

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
num_epochs = 80 #100
learning_rate = 1e-4

get_histograms_from_dataloader(train, vocab_size=vocab_size, plot=True)
net = MusicLSTMNet(vocab_size) # Initialize LSTM network
print(net)

training_loss, validation_loss = train_lstm(net, num_epochs, train, val, vocab_size, encoder_decoder)

# save net
torch.save(net, paths.model_serialized_dir + model_name + '.pt')

# Create params dictionary to be saved with all hyperparameters
params = dict(vocab_size=vocab_size,
              num_epochs=num_epochs,
              lr=learning_rate,
              num_sequences=num_sequences,
              length_per_seq=length_per_seq)

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


#%%
# Compare target y with output from forward pass from first sample of test_set
# TO DO: Implement a SCORE for all test data

for inputs_one_hot, targets_idx in test:
    if True:
        break

outputs = net.forward(inputs_one_hot)
x = test.dataset.inputs[0]._events
y = test.dataset.targets[0]._events

events = []
for output in outputs.detach().numpy():
    label = np.argmax(output)
    events.append(t._encoder_decoder.class_index_to_event(label, events))