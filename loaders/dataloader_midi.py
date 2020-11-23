import math
import note_seq
from note_seq import midi_io
from os import listdir, open
from os.path import isfile, isdir, join
import numpy as np
from utils.tools import timer

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
