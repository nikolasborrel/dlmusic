import math
import note_seq
from music_utils.tokenizer import MusicTokenizer
from loaders.dataset import Dataset
from utils.tools import split_list
from note_seq import midi_io
from os import listdir, open
from os.path import isfile, isdir, join
import numpy as np
from utils.tools import timer

def create_dataset_from_midi(root_dir, note_chunk=10, predict_num_notes=1, print_info=False, recursive=False) -> ([], [], [], MusicTokenizer):
    note_seqs = load_midi_to_seq(root_dir, recursive=False)

    notes_pieces = [split_list(seq.notes,note_chunk,predict_num_notes) for seq in note_seqs]

    t = MusicTokenizer()

    flatten = lambda t: [item for sublist in t for item in sublist]

    notes_sub_pieces = flatten(notes_pieces) # one level resulting in [[...], [...]]
    t.fit_on_notes_pieces(notes_sub_pieces)

    # summarize what was learned
    print(t.note_counts)
    print(t.piece_count)
    print(t.note_index)
    print(t.note_pieces)
    # integer encode documents - TODO
    #encoded_docs = t.texts_to_matrix(docs, mode='count')
    #print(encoded_docs)

    sequences = t.note_seqs_to_sequences(notes_sub_pieces)
    #maxlen = 100
    #test_padded = pad_sequences(test_sequences, maxlen=maxlen)

    print("Testing sequences:\n", sequences)

    training_set, validation_set, test_set = create_datasets(sequences, Dataset)

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

def create_datasets(sequences, dataset_class, p_train=0.8, p_val=0.1, p_test=0.1) -> ([], [], []):
    # Define partition sizes
    num_train = int(len(sequences)*p_train)
    num_val = int(len(sequences)*p_val)
    num_test = int(len(sequences)*p_test)

    # Split sequences into partitions
    sequences_train = sequences[:num_train]
    sequences_val = sequences[num_train:num_train+num_val]
    sequences_test = sequences[-num_test:]

    def get_inputs_targets_from_sequences(sequences):
        # Define empty lists
        inputs, targets = [], []
        
        # Append inputs and targets s.t. both lists contain L-1 words of a sentence of length L
        # but targets are shifted right by one so that we can predict the next word
        for sequence in sequences:
            inputs.append(sequence[:-1])
            targets.append(sequence[1:])
            
        return inputs, targets

    # Get inputs and targets for each partition
    inputs_train, targets_train = get_inputs_targets_from_sequences(sequences_train)
    inputs_val, targets_val = get_inputs_targets_from_sequences(sequences_val)
    inputs_test, targets_test = get_inputs_targets_from_sequences(sequences_test)

    # Create datasets
    training_set = dataset_class(inputs_train, targets_train)
    validation_set = dataset_class(inputs_val, targets_val)
    test_set = dataset_class(inputs_test, targets_test)

    return training_set, validation_set, test_set