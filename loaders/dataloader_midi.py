import math
import note_seq
from music_utils.tokenizer import TokenizerMonophonic
from loaders.dataset import Dataset
from utils.tools import split_list
from note_seq import midi_io
#import pretty_midi as pm # use this to create pm.KeySignature(key_number,time)
from magenta.pipelines.note_sequence_pipelines import TranspositionPipeline
from os import listdir, open
from os.path import isfile, isdir, join
import numpy as np
from utils.tools import timer
import os
from utils.tools import get_index_string
from typing import List, Tuple
import string

from note_seq.melodies_lib import Melody

def create_dataset_from_midi(root_dir, instruments: Tuple[int,int], note_chunk=10, print_info=False, recursive=False) -> ([], [], [], TokenizerMonophonic):
    
    print("Create...")
    sequences = load_midi_to_seq(root_dir, recursive=False)
    
    if len(sequences) == 0:
        raise Exception(f'No midi files loaded')
    
    #print(sequences)

    print("Tokenize...")
    t = TokenizerMonophonic(min_note=60, max_note=72)
    t.add_songs_from_sequences(sequences, instruments)
    
    # summarize what was learned
    #TODO print(t.note_counts)
    print(f'song count: {t.song_count}')

    if t.song_count == 0:
        raise Exception(f'No songs matching instruments {instruments}')

    #print("Testing sequences:\n", sequences)
    
    training_set, validation_set, test_set = create_datasets(t.songs, Dataset)

    if print_info:
        print(f'We have {t.song_count} sentences and {t.vocab_size} unique tokens in our dataset (including UNK).\n')
        #print('The index of \'b\' is', word_to_idx['b'])
        #print(f'The word corresponding to index 1 is \'{idx_to_word[1]}\'')

        print(f'We have {len(training_set)} samples in the training set.')
        print(f'We have {len(validation_set)} samples in the validation set.')
        print(f'We have {len(test_set)} samples in the test set.')

    return training_set, validation_set, test_set, t

def create_datasets(songs: List[Tuple[Melody,Melody]], dataset_class, p_train=0.8, p_val=0.1, p_test=0.1) -> ([], [], []):
    print("create data set...")

    # Define partition sizes
    num_train = int(len(songs)*p_train)
    num_val = int(len(songs)*p_val)
    num_test = int(len(songs)*p_test)

    # Split sequences into partitions
    songs_train = songs[:num_train]
    songs_val = songs[num_train:num_train+num_val]
    songs_test = songs[-num_test:]

    def get_inputs_targets_from_songs(songs: List[Tuple[Melody, Melody]]):
        # Define empty lists
        inputs, targets = [], []
        
        # Append inputs and targets s.t. both lists contain L-1 words of a sentence of length L
        # but targets are shifted right by one so that we can predict the next word
        for song in songs:            
            inputs.append(song[0])
            targets.append(song[1])
            
        return inputs, targets

    # Get inputs and targets for each partition
    inputs_train, targets_train = get_inputs_targets_from_songs(songs_train)
    inputs_val, targets_val = get_inputs_targets_from_songs(songs_val)
    inputs_test, targets_test = get_inputs_targets_from_songs(songs_test)

    # Create datasets
    training_set = dataset_class(inputs_train, targets_train)
    validation_set = dataset_class(inputs_val, targets_val)
    test_set = dataset_class(inputs_test, targets_test)

    if len(training_set) == 0 or len(validation_set) == 0 or len(test_set) == 0:
        raise Exception('Dataset too small to partition into training, validation and test set')

    return training_set, validation_set, test_set

def load_midi_to_seq(root_dir, recursive=False):
    print('Loading...')
    if recursive:
        return convert_midi_to_note_seq(root_dir)
    else:
        files = midi_file_paths_in_dir(root_dir)
        return np.array([midi_io.midi_file_to_note_sequence(f) for f in files])


def convert_midi_to_note_seq(root_dir):
    files = midi_file_paths_in_dir(root_dir)
    dirs = [join(root_dir, f) for f in listdir(root_dir) if isdir(join(root_dir, f))]

    note_seq_arr1 = np.array([])

    for dir in dirs:
        note_seq_arr1 = np.append(note_seq_arr1, convert_midi_to_note_seq(dir))
    
    note_seq_arr2 = np.array([midi_io.midi_file_to_note_sequence(f) for f in files])    
    return np.append(note_seq_arr1, note_seq_arr2)

def midi_file_paths_in_dir(root_dir):
    file_paths = [join(root_dir, f) for f in listdir(root_dir) if isfile(join(root_dir, f))]
    return [f for f in file_paths if f.lower().endswith('.mid') or f.lower().endswith('.midi')]

def open_file(path):
    with open(path, 'r') as f:
        return f.read()