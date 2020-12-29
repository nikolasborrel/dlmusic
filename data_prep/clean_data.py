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