import math
import note_seq
from music_utils.tokenizer import TokenizerMonophonic
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from loaders.dataset import Dataset
from utils.tools import split_list
from note_seq import midi_io
from magenta.pipelines.note_sequence_pipelines import TranspositionPipeline
from os import listdir, open
from os.path import isfile, isdir, join
import numpy as np
from utils.tools import timer
import os
from utils.tools import get_index_string
from typing import List, Tuple, Dict
import string
import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from note_seq.melodies_lib import Melody

# https://pytorch.org/docs/stable/data.html
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

# Global dictionaries: their values can be overwritten locally according to corresponding kwargs 
tokenizer_kwargs = {'split_in_bar_chunks': 4, 'steps_per_quarter': 1, 'min_note': 60, 'max_note':72}

dataset_split_kwargs = {'p_train': 0.8, 'p_val': 0.1, 'p_test': 0.1,
                        'batch_size': 1, 'eval_batch_size': 1}



@timer
def create_dataset_from_midi(root_dir, 
                             name_instr_lead: Tuple[str,int], 
                             name_instr_accomp: Tuple[str,int], 
                             print_info=False, 
                             recursive=False,
                             **kwargs) -> (DataLoader, DataLoader, DataLoader, TokenizerMonophonic):
    """
    kwargs: key-word arguments of dataset_split_kwargs and tokenizer_kwargs
            if some keys are absent the default initialized at the start of the script will be used
    """

    # Split kwargs into tokenizer_kwargs and dataset_split_kwargs and update them
    # tokenizer_kwargs
    common_kwargs = dict(filter(lambda elem: elem[0] in tokenizer_kwargs.keys(), kwargs.items()))
    tokenizer_kwargs.update(common_kwargs)

    # dataset_split_kwargs
    common_kwargs = dict(filter(lambda elem: elem[0] in dataset_split_kwargs.keys(), kwargs.items()))
    dataset_split_kwargs.update(common_kwargs)
        
    print("Create...")
    extract_names = (name_instr_lead[0], name_instr_accomp[0])
    extract_instruments = [name_instr_lead[1], name_instr_accomp[1]]

    name_instrument_map = {
        extract_names[0]:  extract_instruments[0], 
        extract_names[1]:  extract_instruments[1]
        }

    sequences = load_midi_to_seq(root_dir, name_instrument_map, recursive=False)
    
    if len(sequences) == 0:
        raise Exception(f'No midi files loaded')

    print("Tokenize...")
    t = TokenizerMonophonic(**tokenizer_kwargs)
    t.add_songs(sequences, extract_instruments)
    
    # summarize what was learned
    print(f'song count: {t.song_count}')

    if t.song_count == 0:
        raise Exception(f'No songs matching instruments {extract_instruments}')
    
    training_set, validation_set, test_set = create_datasets(t, Dataset, **dataset_split_kwargs)
    if print_info:
        print(f'We have {t.song_count} sentences and {t.vocab_size} unique tokens in our dataset (including NO_EVENT = 0 and NOTE_OFF = 1).\n')
        #print(f'We have {t._split_in_bar_chunks * 16} events in each sentence')
        print(f'We have {len(training_set)} samples in the training set.')
        print(f'We have {len(validation_set)} samples in the validation set.')
        print(f'We have {len(test_set)} samples in the test set.')

    return training_set, validation_set, test_set, t

def encode_from_midi(root_dir, name_instr: Tuple[str,int], 
    max_bars_chunk, print_info=False, recursive=False) -> ([], [], [], TokenizerMonophonic):
    
    print("Create...")
    extract_instruments = [name_instr[1]]
    name_instrument_map = { name_instr[0]:  name_instr[1] }

    sequences = load_midi_to_seq(root_dir, name_instrument_map, recursive=False)
    
    if len(sequences) == 0:
        raise Exception(f'No midi files loaded')

    print("Tokenize...")
    t = TokenizerMonophonic(split_in_bar_chunks=max_bars_chunk, min_note=60, max_note=72)
    t.add_songs(sequences, extract_instruments)
    
    # summarize what was learned
    print(f'song count: {t.song_count}')

    if t.song_count == 0:
        raise Exception(f'No songs matching instruments {extract_instruments}')
    
    return t

## dataset_split_kwargs = p_train=0.8, p_val=0.1, p_test=0.1, batch_size=1
def create_datasets(t: TokenizerMonophonic, dataset_class, p_train=0.8, p_val=0.1, p_test=0.1, \
                    batch_size=1, eval_batch_size=1) -> (DataLoader, DataLoader, DataLoader):
    """
    key-word arguments are dataset_split_kwargs
    """
    print("create data set...")
    songs = list(zip(t.song_parts_lead, t.song_parts_accomp))

    batch_size = 1 #16
    eval_batch_size = 1 #16

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
    training_set = dataset_class(inputs_train, targets_train, t.encoder_decoder)
    validation_set = dataset_class(inputs_val, targets_val, t.encoder_decoder)
    test_set = dataset_class(inputs_test, targets_test, t.encoder_decoder)

    if len(training_set) == 0 or len(validation_set) == 0 or len(test_set) == 0:
        raise Exception('Dataset too small to partition into training, validation and test set')
    
    # The loaders perform the actual work
    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_set, batch_size=eval_batch_size, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_set, batch_size=eval_batch_size, shuffle=True, num_workers=0)

    return train_loader, validation_loader, test_loader

def load_midi_to_seq(root_dir, name_instrument_map, recursive=False):
    print('Loading...')
    if recursive:
        return convert_midi_to_note_seq(root_dir, name_instrument_map)
    else:
        files = midi_file_paths_in_dir(root_dir)
        return np.array([midi_io.midi_file_to_note_sequence(f, name_instrument_map) for f in files])


def convert_midi_to_note_seq(root_dir, name_instrument_map):
    files = midi_file_paths_in_dir(root_dir)
    dirs = [join(root_dir, f) for f in listdir(root_dir) if isdir(join(root_dir, f))]

    note_seq_arr1 = np.array([])

    for dir in dirs:
        note_seq_arr1 = np.append(note_seq_arr1, convert_midi_to_note_seq(dir, name_instrument_map))
    
    note_seq_arr2 = np.array([midi_io.midi_file_to_note_sequence(f, name_instrument_map) for f in files])    
    return np.append(note_seq_arr1, note_seq_arr2)

def midi_file_paths_in_dir(root_dir):
    file_paths = [join(root_dir, f) for f in listdir(root_dir) if isfile(join(root_dir, f))]
    return [f for f in file_paths if f.lower().endswith('.mid') or f.lower().endswith('.midi')]

def open_file(path):
    with open(path, 'r') as f:
        return f.read()