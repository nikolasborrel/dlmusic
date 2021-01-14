import sys
sys.path.append('../note_seq') # needed unless installing forked lib from github

# loading/saving pyTorch models: https://pytorch.org/tutorials/beginner/saving_loading_models.html

import numpy as np
import matplotlib.pyplot as plt
from models.model_lstm import MusicLSTMNet
from loaders.dataloader_midi import encode_from_midi, tokenizer_kwargs
from evaluation.eval_lstm import eval_lstm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils.paths as paths
from utils.tools import flatten
from note_seq import midi_io
import json

# Initialize folder and name of model to read results and parameters
model_name = 'midi_epochs_50_1_hidden_300.pt'
results_folder_path = paths.model_serialized_dir
path_to_model = results_folder_path + model_name
path_to_params = results_folder_path + model_name[:-3] + '.json'


path_input_melody  = paths.root_dir + '/dlmusic_data/midi_data_out/mel_input/'
path_output_melody = paths.root_dir + '/dlmusic_data/midi_data_out/learned/'

# Set seed such that we always get the same dataset
np.random.seed(42)

with open(path_to_params) as json_file:
    params = json.load(json_file)
common_kwargs = dict(filter(lambda elem: elem[0] in tokenizer_kwargs.keys(),params.items()))
tokenizer_kwargs.update(common_kwargs)

instruments = [0]
lead_instrument   = ('melody',instruments[0])

tokenizer = encode_from_midi(path_input_melody, lead_instrument, print_info=True, **tokenizer_kwargs)
encoder_decoder = tokenizer.encoder_decoder
num_sequences = tokenizer.song_count
vocab_size = tokenizer.vocab_size

input_song_parts = tokenizer.song_parts_lead

# Initialize a LSTM network
net = torch.load(path_to_model)
events = eval_lstm(net, input_song_parts, vocab_size, encoder_decoder)

tokenizer.to_midi(events, path_output_melody, filename=model_name[:-3] + '_out.mid')
input_events = [mel._events for mel in input_song_parts]
#midi_io.sequence_proto_to_midi_file(, path_output_melody + 'input.mid')