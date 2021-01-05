import sys
sys.path.append('../note_seq') # needed unless installing forked lib from github

import numpy as np
import matplotlib.pyplot as plt
from models.model_lstm import MusicLSTMNet, LSTM
from loaders.dataloader_midi import create_dataset_from_midi
from training.train_lstm import train_lstm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils.paths as paths
from utils.tools import flatten
from collections import Counter
from models.distributions import get_histograms_from_dataloader

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
vocab_size = t.vocab_size

mel_notes, bass_notes = get_histograms_from_dataloader(train, vocab_size=vocab_size, \
                                                       plot=True)
print('Melody histogram: ', Counter(mel_notes))
print('Bass histogram: ', Counter(bass_notes))
