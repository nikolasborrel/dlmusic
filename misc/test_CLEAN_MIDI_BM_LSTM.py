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
import time
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


#%%

from collections import OrderedDict
from music_utils.notes import midi_number_to_note
import note_seq
from note_seq import melodies_lib
from note_seq import sequences_lib
from note_seq.melodies_lib import Melody
from note_seq import encoder_decoder
from note_seq import melody_encoder_decoder
from note_seq import midi_io
from note_seq.protobuf import music_pb2
from note_seq.constants import DEFAULT_QUARTERS_PER_MINUTE
from note_seq.protobuf.generator_pb2 import GeneratorOptions
from note_seq.protobuf.music_pb2 import NoteSequence
from typing import Tuple, List
import note_seq

class TokenizerMonophonic():
    
    DEFAULT_MIN_NOTE = 48
    DEFAULT_MAX_NOTE = 84

    #min=60 max=72 -> one octave    
    def __init__(self, min_note=DEFAULT_MIN_NOTE, max_note=DEFAULT_MAX_NOTE) -> None :
        # self._note_counts = OrderedDict() 
        # self._note_pieces = {} 
        # self._note_index = {}
        # self._index_note = {}
        self._songs = []
        self._song_count = 0
        self._min_note = min_note
        self._max_note = max_note

        self._encoder_decoder = encoder_decoder.OneHotEventSequenceEncoderDecoder(
            melody_encoder_decoder.MelodyOneHotEncoding(min_note, max_note)) # min_note=DEFAULT_MIN_NOTE, max_note=DEFAULT_MAX_NOTE

        # Additional labels are NO_EVENT = 0 and NOTE_OFF = 1
        assert(self._encoder_decoder.input_size, max_note - min_note + 2) 
        assert(self._encoder_decoder.num_classes, max_note - min_note + 2)


    def add_songs_from_sequences(self, songs: List[NoteSequence], instruments: Tuple[int, int], steps_per_quarter=4, ignore_polyphonic_notes=True):
        '''
        sequences is a list of list of note_seq, each inner list corresponding to a song
        '''
        self.loading_errors = []
        for sequence in songs:      
            try:
                quantized_sequence = sequences_lib.quantize_note_sequence(
                    sequence, steps_per_quarter=steps_per_quarter)
            except Exception as e:
                self.loading_errors.append((sequence.filename, e))
            print(loading_errors)

            # EXTRACT FIRST INSTRUMENT
            melody0 = Melody()
            melody0.from_quantized_sequence(
                quantized_sequence, instrument=instruments[0], ignore_polyphonic_notes=ignore_polyphonic_notes)
            
            transpose_to_key = 0        

            # squeeze midi into octaves determined by min_note and max_note and transposes to key = 0 => C major / A minor
            melody0.squash(
                self._min_note,
                self._max_note,
                transpose_to_key)

            # EXTRACT SECOUND INSTRUMENT
            melody1 = Melody()
            melody1.from_quantized_sequence(
                quantized_sequence, instrument=instruments[1], ignore_polyphonic_notes=ignore_polyphonic_notes)
            
            transpose_to_key = 0        

            # squeeze midi into octaves determined by min_note and max_note and transposes to key = 0 => C major / A minor
            melody1.squash(
                self._min_note,
                self._max_note,
                transpose_to_key)

            if len(melody0) > 0 and len(melody1) > 0:
                self._song_count = self._song_count + 1
                self._songs.append((melody0, melody1))

    @property
    def songs(self):
        return self._songs

    # The encoder/decoder for a monophonic sequence
    @property
    def encoder_decoder(self):
        return self._encoder_decoder

    # # A (ordered) dictionary of words and their counts.
    # @property
    # def note_counts(self):
    #     return self._note_counts

    # # A dictionary of words and how many documents each appeared in.
    # @property
    # def note_pieces(self):
    #     return self._note_pieces

    # # A dictionary of notes and their uniquely assigned integers.
    # @property
    # def note_index(self):
    #     return self._note_index

    # @property
    # def index_note(self):
    #     return self._index_note

    # An integer count of the total number of documents that were used to fit the Tokenizer.
    @property
    def song_count(self):
        return self._song_count

    @property
    def vocab_size(self):
        return self._encoder_decoder.num_classes

#%%

folder_name = '/home/sarantos/Documents/Music_AI/CLEAN_MIDI_BM'
midis = read_midis_from_clean_midi_BM(folder=folder_name, read_midi=midi_io.midi_file_to_note_sequence)

#%%
sequences = midis[0:30]
instruments = (0,1)
print("Tokenize...")
t = TokenizerMonophonic(min_note=60, max_note=72)
t.add_songs_from_sequences(sequences, instruments)

# summarize what was learned
#TODO print(t.note_counts)
print(f'song count: {t.song_count}')

if t.song_count == 0:
    raise Exception(f'No songs matching instruments {instruments}')

#print("Testing sequences:\n", sequences)

from loaders.dataloader_midi import create_datasets
from loaders.dataset import Dataset


training_set, validation_set, test_set = create_datasets(t.songs, Dataset)


print(f'We have {t.song_count} sentences and {t.vocab_size} unique tokens in our dataset (including UNK).\n')
#print('The index of \'b\' is', word_to_idx['b'])
#print(f'The word corresponding to index 1 is \'{idx_to_word[1]}\'')

print(f'We have {len(training_set)} samples in the training set.')
print(f'We have {len(validation_set)} samples in the validation set.')
print(f'We have {len(test_set)} samples in the test set.')


#%%

from note_seq import encoder_decoder
encoder_decoder = encoder_decoder
num_sequences = t.song_count
vocab_size = t.vocab_size

#%%
# Test length of input target

x = training_set.inputs[0]
y = training_set.targets[0]

x1 = x.to_sequence()
y1 = y.to_sequence()

from loaders.dataloader_midi import get_instruments_from_NoteSequence

note_seq.sequence_proto_to_midi_file(x1, 'input_mel.mid')
note_seq.sequence_proto_to_midi_file(y1, 'target_bass.mid')

#%%
# Set seed
np.random.seed(42)

path_to_midi_dir = folder_name

#instruments_to_extract = (54, 34) # voice and bass
instruments_to_extract = (0, 1) # voice and bass

# Hyper-parameters
num_epochs = 200
training_set, validation_set, test_set, tokenizer \
    = create_dataset_from_midi(path_to_midi_dir, instruments_to_extract, print_info=True)

#%%
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



