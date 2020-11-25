#%%
import numpy as np
import pandas as pd
import scipy as sp

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

import json
import os
import inspect
import collections
import bisect

import time
import types
import sys
import re
from collections import abc, OrderedDict
#from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

flatten = lambda l: [item for sublist in l for item in sublist]
##############################################################################
######################### TIMERS AND BENCHMARK ###############################
##############################################################################
def timer(orig_func):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = orig_func(*args, **kwargs)
        t2 = time.time() - t1
        print('{} ran in: {} sec'.format(orig_func.__name__, t2))
        return result
    print('Starting: {}'.format(orig_func.__name__))
    return wrapper


def call_counter(func):
    """
    Add as a decorator to a function @call_counter
    Access locally the number of time being called as func.calls
    """
    def helper(*args, **kwargs):
        helper.calls += 1
        return func(*args, **kwargs)
    helper.calls = 0
    helper.__name__= func.__name__
    return helper

##############################################################################
######################### JSON    ############################################
##############################################################################

def read_json(filepath, extension):
    '''
    :filepath : string of path ending with /
    :extension : string
    :returns : data
    '''
    if filepath[-1] == '/':
        with open(filepath + extension) as json_file:
            data = json.load(json_file)
    else:
        with open(filepath + '/' + extension) as json_file:
            data = json.load(json_file)
    return data


def write_json(data, filename):
    '''
    :data: object to be saved as json
    :filename: string name of json file
    '''
    with open(filename, 'w') as f:
        json.dump(data, f)


def write_list_as_csv_txt(filename, some_list):
    with open(filename,'w') as g:
        for count, idx in enumerate(some_list):
            if count == len(some_list) - 1:
                g.write(str(idx))
            else:
                g.write(str(idx) + ',')
    return
##############################################################################
######################### Dictionary Ops       ###############################
##############################################################################


def get_index_string(text, string):
    ''' Gets indexes of regex string pattern found in text'''
    index = []
    for m in re.finditer(string, text):
        #print(string + 'found between ', m.start(), m.end())
        val = m.start()
        index.append(val)
    return index


def order_dict_keys(x):
    """
    x :  dict

    od : x with ordered  keys

    """
    od = {}
    keys = list(x.keys())
    ids = sorted([int(v) for v in keys])
    for key in ids:
        od[key] = x[key]
    return od

class AutoVivification(dict):
    """Implementation of perl's autovivification feature."""
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value

##############################################################################
######################### Animations            ###############################
##############################################################################

def anim_im(image_data, n1=None, n2=None):
    '''
    Animation of a sitk or numpy image
    Parameters
    ----------
    image_data : numpy array.
    n1 : index of slice to begin
    n2 : index of slice to end

    Returns
    -------
    ani : matplotlib animation object
    '''
    assert len(image_data.shape) == 3

    fig = plt.figure()
    ims = []
    #a = np.argmin(np.shape(image_data))

    if n1 is not None:
        for i in range(n1,n2 + 1):
            im = plt.imshow(image_data[i,:,:])
            ims.append([im])
        fig.colorbar(im)
        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                            repeat_delay=1000)
        return ani
    else:
        #for i in range(0, np.shape(image_data)[0]):
        for i in range(0, np.shape(image_data)[2]):

            im = plt.imshow(image_data[:,:,i], interpolation='none',animated=True)

            ims.append([im])
        fig.colorbar(im)
        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                        repeat_delay=1000)
        return ani


############# COORDINATES #############################################
def knnsearch(x, y):
    '''
    Finds indexes of x that are closest to values of y 
    and corresponding distances
    '''
    if isinstance(y, (float, int)):
        y = [y]
    x = np.asarray(x)
    y = np.asarray(y)
    indexes = []
    distances = []
    for y_val in y:
        dists = np.abs(x -  y_val * np.ones(len(x)))
        idx = dists.argmin()
        dist_val = x[idx] - y_val
        indexes.append(idx)
        distances.append(dist_val)
        
    return indexes, distances


########################## AUDIO ########################################
def rec_audio(fs, t_end, filename='output.wav'):
    import sounddevice as sd
    from scipy.io.wavfile import write
    print('starts_recording')
    myrecording = sd.rec(int(t_end * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    write(filename, fs, myrecording)  # Save as WAV file 

    
def audio_to_np(filename):
    from scipy.io import wavfile 
    a = wavfile.read(filename)
    fs = a[0]
    if len(a) > 1:
        print('stereo')
        x = a[1].sum(axis=1) / 2
    #x = np.array([a[1], dtype=float])
    
    x = x/max(abs(x))
    print(x.dtype)
    return x, fs

############ MUSIC ######################################################

def equal_temperament_scale(n_octaves=None):
    scale = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    notes = []
    if n_octaves == None:
        n_octaves = 9
    for n in range(0,n_octaves):
        octave = [note + str(n) for note in scale]
        notes += octave
    f_ref = 440 #A4
    notes_freq = []
    for i in range(0, len(notes)):
        val = f_ref * 2 ** ((- notes.index('A4') + i) / 12)
        notes_freq.append(val)
    return notes, notes_freq


def find_closest_note(notes, notes_freq, f):
    if not isinstance(f, list):
        f = [f]
    indexes, distances = knnsearch(notes_freq, f)
    note = notes[indexes[0]]
    print('Closest note is {} with distance {} Hz'.format(note, distances[0]))
    return note

def split_list(list_, chunk_size, overlap):
    if chunk_size < 1:
        raise Exception("chunk size too small")
    if overlap >= chunk_size:
        raise Exception("overlap too large")

    return [list_[i:i+chunk_size] for i in range(0, len(list_)-chunk_size+1, chunk_size-overlap)]

def pad_sequences(sequences: [[]], maxlen) -> [[]]:
    seqs_padded = [[0 for col in range(maxlen)] for row in range(len(sequences))]
    for i in range(0, len(sequences)-1):
        if len(sequences) < maxlen:
            seqs_padded[i,0:len(sequences)-1] = sequences
        else:
            seqs_padded[i,:] = sequences[0:maxlen-1]

    return seqs_padded
