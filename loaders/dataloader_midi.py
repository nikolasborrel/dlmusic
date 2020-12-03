import math
import note_seq
from music_utils.tokenizer import MusicTokenizer
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

def create_dataset_from_midi(root_dir, note_chunk=10, num_pred=1, print_info=False, recursive=False) -> ([], [], [], MusicTokenizer):
    note_seqs = load_midi_to_seq(root_dir, recursive=False)
    
    note_seqs = transpose_note_seqs_to_c(note_seqs)

    notes_pieces = [split_list(seq.notes,note_chunk,num_pred) for seq in note_seqs]

    t = MusicTokenizer()

    flatten = lambda t: [item for sublist in t for item in sublist]

    notes_sub_pieces = flatten(notes_pieces) # one level resulting in [[...], [...]]
    t.fit_on_notes_pieces(notes_sub_pieces)

    # summarize what was learned
    print(t.note_counts)
    print(t.piece_count)
    #print(t.note_index)
    #print(t.note_pieces)
    # integer encode documents - TODO
    #encoded_docs = t.texts_to_matrix(docs, mode='count')
    #print(encoded_docs)

    sequences = t.note_seqs_to_sequences(notes_sub_pieces)
    #test_padded = pad_sequences(test_sequences, maxlen=100)

    #print("Testing sequences:\n", sequences)

    training_set, validation_set, test_set = create_datasets(sequences, Dataset, num_pred=num_pred)

    if print_info:
        print(f'We have {t.piece_count} sentences and {t.vocab_size} unique tokens in our dataset (including UNK).\n')
        #print('The index of \'b\' is', word_to_idx['b'])
        #print(f'The word corresponding to index 1 is \'{idx_to_word[1]}\'')

        print(f'We have {len(training_set)} samples in the training set.')
        print(f'We have {len(validation_set)} samples in the validation set.')
        print(f'We have {len(test_set)} samples in the test set.')

    return training_set, validation_set, test_set, t

@timer
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

def create_datasets(sequences, dataset_class, num_pred=1, p_train=0.8, p_val=0.1, p_test=0.1) -> ([], [], []):
    # Define partition sizes
    num_train = int(len(sequences)*p_train)
    num_val = int(len(sequences)*p_val)
    num_test = int(len(sequences)*p_test)

    # Split sequences into partitions
    sequences_train = sequences[:num_train]
    sequences_val = sequences[num_train:num_train+num_val]
    sequences_test = sequences[-num_test:]

    def get_inputs_targets_from_sequences(sequences, num_pred):
        # Define empty lists
        inputs, targets = [], []
        
        # Append inputs and targets s.t. both lists contain L-1 words of a sentence of length L
        # but targets are shifted right by one so that we can predict the next word
        for sequence in sequences:
            inputs.append(sequence[:-num_pred])
            targets.append(sequence[num_pred:])
            
        return inputs, targets

    # Get inputs and targets for each partition
    inputs_train, targets_train = get_inputs_targets_from_sequences(sequences_train, num_pred)
    inputs_val, targets_val = get_inputs_targets_from_sequences(sequences_val, num_pred)
    inputs_test, targets_test = get_inputs_targets_from_sequences(sequences_test, num_pred)

    # Create datasets
    training_set = dataset_class(inputs_train, targets_train)
    validation_set = dataset_class(inputs_val, targets_val)
    test_set = dataset_class(inputs_test, targets_test)

    return training_set, validation_set, test_set

def transpose_note_seqs_to_c(note_seqs):
    note_seqs_transposed = []
    for k in note_seqs:
        if len(k.key_signatures) > 1:
            print("WARNING: more than one key signatures were found - only the first signature is used.")

        if len(k.key_signatures) > 0:
            transpose_interval = -k.key_signatures[0].key
            tp = TranspositionPipeline([transpose_interval])
            k_transformed = tp.transform(k)[0]            
            # TODO: create datastructure to keep original key as well, to be able to transpose back if needed
            # set FIRST signature to key of C. 
            k_transformed.key_signatures[0].key = 0 # NOTE: printing is empty for C=0, 
            note_seqs_transposed.append(k_transformed)
        else:
            note_seqs_transposed.append(k)
    
    return note_seqs_transposed

@timer
def remove_files_from_clean_midi(dataset_dir, 
                                 midi_read=midi_io.midi_file_to_note_sequence):
    """
    Parameters
    ----------
    dataset_dir : directory path of dataset
    midi_read : function to read midi
                The default is midi_io.midi_file_to_note_sequence.
    
    Returns
    ----------
    del_files: str of artist name and song name deleted
    n_files: total number of midi files before removal
    errors: set of all errors for the files deleted
    """
    errors = set()
    del_files = []
    del_art = []
    n_files = 0
    for root, subdirs, files in os.walk(dataset_dir):
        if len(files) > 0:
            art_name = root[get_index_string(root,'/')[-1] + 1:]
            print(art_name)
            count = 0
            for file in files:
                if not (file.lower().endswith('.mid') or file.lower().endswith('.midi')):
                    continue
                n_files += 1
                filepath = root + '/' + file
                try:
                    k = midi_read(filepath)
                except Exception as e:
                    if 'KeyboardInterrupt' in str(e):
                        print('Keyboard Interrupt, file {} not removed'.format(file))
                        return del_files, n_files, errors
                    print(e)
                    count += 1
                    errors.add(e)
                    print('From {} file: {} is deleted'.format(art_name, file))
                    os.remove(filepath)
                    #del_files.append((art_name + '/' + file, *e))
                    del_files.append(art_name+'/' + file)
                    #if len(files) == count:
                    #    os.rmdir(root)
                    #    del_art.append(art_name)
                    #    print('Removing {} folder'.format())
                            
    print('{} files deleted out of {}'.format(len(del_files), n_files))
    #print('{} Artists deleted'.format(len(del_art)))
    return del_files, n_files, errors


def get_artists_midi_dict_generator(dataset_dir, 
                                  midi_read=midi_io.midi_file_to_note_sequence):
    """
    Parameters
    ----------
    dataset_dir : directory path of dataset
    midi_read : function to read midi
                The default is midi_io.midi_file_to_note_sequence.

    """
    art_midi = {} #Dictionary where keys: artist names, vals: generator object of midi_read func
    art_nfiles = {}
    for root, subdirs, files in os.walk(dataset_dir):
        if len(files) > 0:
            art_name = root[get_index_string(root,'/')[-1] + 1:]
            print(art_name)
            count = 0
            filepaths = [root + '/' + f for f in files if (f.lower().endswith('.mid') or f.lower().endswith('.midi'))]
            art_midi[art_name] = map(midi_read, filepaths)
            art_nfiles[art_name] = len(filepaths)
    return art_midi, art_nfiles

