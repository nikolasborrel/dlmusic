import sys
sys.path.append('../note_seq') # needed unless installing forked lib from github

# loading/saving pyTorch models: https://pytorch.org/tutorials/beginner/saving_loading_models.html

import numpy as np
import matplotlib.pyplot as plt
from models.model_lstm import MusicLSTMNet
from loaders.dataloader_midi import encode_from_midi
from evaluation.eval_lstm import eval_lstm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils.paths as paths
from utils.tools import flatten
from note_seq import midi_io

model_name = 'music_lstm.pt'
path_to_model = paths.model_serialized_dir + model_name

path_input_melody  = paths.root_dir + '/dlmusic_data/midi_data_out/mel_input/'
path_output_melody = paths.root_dir + '/dlmusic_data/midi_data_out/learned/'

# Set seed such that we always get the same dataset
np.random.seed(42)

instruments = [0]
lead_instrument   = ('melody',instruments[0])

max_bars = 16

tokenizer = encode_from_midi(path_input_melody, lead_instrument, max_bars, print_info=True)
encoder_decoder = tokenizer.encoder_decoder
num_sequences = tokenizer.song_count
vocab_size = tokenizer.vocab_size

input_song_parts = tokenizer.song_parts_lead

# Initialize a LSTM network
net = torch.load(path_to_model)
events = eval_lstm(net, input_song_parts, vocab_size, encoder_decoder)

tokenizer.to_midi(events, path_output_melody, filename='out.mid')
midi_io.sequence_proto_to_midi_file(input_song_parts[0].to_sequence(), path_output_melody + 'input.mid')