#%%
import numpy as np
from music_utils.notes import get_midi_pitch_to_note_names_dict
from note_seq import midi_io
#from note_seq.protobuf import music_pb2
import pretty_midi as pm
from utils.tools import read_json
#import loaders.dataloader_midi as loader
from note_seq.protobuf.music_pb2 import NoteSequence
from loaders.dataloader_midi import remove_files_from_clean_midi, get_artists_midi_dict_generator, get_instruments_from_NoteSequence
from loaders.dataloader_midi import read_midis_from_clean_midi_BM
# https://github.com/magenta/magenta/blob/master/magenta/pipelines/note_sequence_pipelines_test.py
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import shutil
import note_seq
from utils.tools import write_list_as_csv_txt
import copy
#%%
import numpy as np
import matplotlib.pyplot as plt
from models.model_lstm import MyRecurrentNet
from loaders.dataloader_midi import create_dataset_from_midi
from training.train_lstm import train_lstm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


folder_name = '/home/sarantos/Documents/Music_AI/CLEAN_MIDI_BM'
midis = read_midis_from_clean_midi_BM(folder=folder_name, read_midi=midi_io.midi_file_to_note_sequence)


# Set seed
np.random.seed(42)

path_to_midi_dir = folder_name

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



