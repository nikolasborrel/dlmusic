import numpy as np
from music_utils.notes import get_midi_pitch_to_note_names_dict
from note_seq import midi_io
#from note_seq.protobuf import music_pb2
import pretty_midi as pm
from utils.tools import read_json, Timer
#import loaders.dataloader_midi as loader
from note_seq.protobuf.music_pb2 import NoteSequence
from loaders.dataloader_midi import remove_files_from_clean_midi, get_artists_midi_dict_generator

# https://github.com/magenta/magenta/blob/master/magenta/pipelines/note_sequence_pipelines_test.py
import matplotlib.pyplot as plt
from tqdm import tqdm


#CHANGE ME
dataset_dir = '/home/sarantos/Documents/Music_AI/clean_midi'

#%%

'''
Remove files that cannot be transformed to a NoteSequence midi object
Only run it once
Should delete 276 files out of 17256 (Function should take around 30 minutes or so)
'''
del_files, n_files, errors = remove_files_from_clean_midi(dataset_dir)
#

#%%
#Get Dataset Dict with values a generator map
art_midi, art_nfiles = get_artists_midi_dict_generator(dataset_dir)

mean_nfiles = np.mean(list(art_nfiles.values()))
print(f'Mean number of files per artist: {mean_nfiles}')
plt.plot(list(art_nfiles.values()))
plt.xlabel('Artist ID')
plt.ylabel('Number of Songs')
plt.title('Songs per Artist')

# Artists with songs more than value thres
thres = 100
keys = [key  for (key, val) in art_nfiles.items() if val > thres]
print(f'Artists with songs more than {thres}: {keys}')
max_nfiles = max(art_nfiles.values())
#Beatles is theartist with most files 773

plt.figure()
plt.hist(list(art_nfiles.values()), bins=np.arange(1,max_nfiles+1, 1), density=True)
plt.title('Histogram of number of songs per artist')
#%%
import random
#From the Histogram we can see that most of the data is with artists with less than 6 songs
#Select 3 random artists that we only have 2 files
keys = [key  for (key, val) in art_nfiles.items() if val == 2]

n_picks = 3
idxs  = [random.randint(0,len(keys)) for _ in range(0,n_picks)]

songs = []

artists = list(map(keys.__getitem__, idxs))
print(f'Randomly selected artists {artists}')

for artist in artists:
    songs.append(list(art_midi[artist]))


def get_instruments_from_NoteSequence(k):
    unique_instruments = set([m.instrument for m in k.notes])
    instruments = {id: inst.name.rstrip() for id, inst in zip(range(0, len(unique_instruments)),
                                            k.instrument_infos)}
    bass_idxs = []
    melody_idxs = []
    for inst_id, instrument in instruments.items():
        if 'bass' in instrument.lower():
            bass_idxs.append(inst_id)
            
        if 'melody' in instrument.lower():
            melody_idxs.append(inst_id)
    return instruments, bass_idxs, melody_idxs
    
from utils.tools import flatten

for k in flatten(songs):
    instruments, bass_idxs, melody_idxs = get_instruments_from_NoteSequence(k)
    print(instruments.values())
    print('bass: ', bass_idxs)
    print('melody: ', melody_idxs) 

#%%

from utils.tools import get_index_string
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
    art_songs = {}
    for root, subdirs, files in os.walk(dataset_dir):
        if len(files) > 0:
            art_name = root[get_index_string(root,'/')[-1] + 1:]
            print(art_name)
            count = 0
            filepaths = [root + '/' + f for f in files if (f.lower().endswith('.mid') or f.lower().endswith('.midi'))]
            art_midi[art_name] = map(midi_read, filepaths)
            art_nfiles[art_name] = len(filepaths)
            songs = list(map(lambda x : x[len(root + '/'):-4], filepaths))
            art_songs[art_name] = songs
    return art_midi, art_nfiles, art_songs
art_midi, art_nfiles, art_songs = get_artists_midi_dict_generator(dataset_dir)


#%%
from utils.tools import AutoVivification
import json
filename = 'clean_midi.json'

count = 0 

with open(filename, 'w') as f:
    for name, gen in tqdm(art_midi.items()):
        D = AutoVivification()
        D[name] = {}
        tunes = list(gen)
        for idx, k in enumerate(tunes):
            #print(len(k))
            try: 
                song = art_songs[name][idx]
                D[name][song] = {}
                unique_instruments = list(set([m.instrument for m in k.notes]))
                D[name][song]['unique_instruments'] = unique_instruments
                D[name][song]['instruments'] = {id: inst.name.rstrip() for id, inst in zip(range(0, len(unique_instruments)),
                                            k.instrument_infos)}
            except Exception as e:
                print(e)
                continue
        json.dump(D, f)


#%%
        
from utils.tools import AutoVivification
import json
filename = 'clean_midi.json'

count = 0 
data = AutoVivification()

for name, gen in tqdm(art_midi.items()):
   
    data[name] = {}
    tunes = list(gen)
    for idx, k in enumerate(tunes):
        #print(len(k))
        try: 
            song = art_songs[name][idx]
            data[name][song] = {}
            unique_instruments = list(set([m.instrument for m in k.notes]))
            data[name][song]['unique_instruments'] = unique_instruments
            data[name][song]['instruments'] = {id: inst.name.rstrip() for id, inst in zip(range(0, len(unique_instruments)),
                                        k.instrument_infos)}
        except Exception as e:
            print(e)
            continue


#%%
            
all_names = set()
filepaths = []
for artist, songs in data.items():
    for tune in songs:
        all_names.add(tune[])
    
            
unique_names = 


#%%
        
with open(os.getcwd() + '/' + filename, 'r') as json_file:
    D = json.load(json_file)

tweets = []
for line in open(filename, 'r'):
    tweets.append(json.loads(line))

#%%
keys = [key  for (key, val) in art_nfiles.items() if val == 2]

n_picks = 3
idxs  = [random.randint(0,len(keys)) for _ in range(0,n_picks)]

songs = []

artists = list(map(keys.__getitem__, idxs))
print(f'Randomly selected artists {artists}')

for artist in artists:
    songs.append(list(art_midi[artist]))
#%%
with Timer('List generator'):
    D = {key: list(songs)  for key, songs in tqdm(art_midi.items())}

#%%
with open(filename, 'w') as f:
    D = {}
    D['erato'] = {}
    D['erato']['a'] = 2
    json.dump(D,f)


class MidiDataset():
    def __init__(self,filepath, artists, songs):
        self.filepath = filepath
        self.artists = artists
        self.songs = songs
        

class MidiSong():
    def __init__(self):
        
    
#%%
        
from utils.tools import get_index_string
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
    art_songs = {}
    for root, subdirs, files in os.walk(dataset_dir):
        if len(files) > 0:
            art_name = root[get_index_string(root,'/')[-1] + 1:]
            print(art_name)
            count = 0
            filepaths = [root + '/' + f for f in files if (f.lower().endswith('.mid') or f.lower().endswith('.midi'))]
            art_midi[art_name] = map(midi_read, filepaths)
            art_nfiles[art_name] = len(filepaths)
            songs = list(map(lambda x : x[len(root + '/'):-4], filepaths))
            art_songs[art_name] = songs
    return art_midi, art_nfiles, art_songs
art_midi, art_nfiles, art_songs = get_artists_midi_dict_generator(dataset_dir)

        
from utils.tools import AutoVivification
import json
filename = 'clean_midi.json'

def get_instruments_from_NoteSequence(k):
    unique_instruments = set([m.instrument for m in k.notes])
    instruments = {id: inst.name.rstrip() for id, inst in zip(range(0, len(unique_instruments)),
                                            k.instrument_infos)}
    bass_idxs = []
    melody_idxs = []
    for inst_id, instrument in instruments.items():
        if 'bass' in instrument.lower():
            bass_idxs.append(inst_id)
            
        if 'melody' in instrument.lower():
            melody_idxs.append(inst_id)
    return instruments, bass_idxs, melody_idxs

all_instruments = set()
filepaths = []
for name, gen in tqdm(art_midi.items()):
    tunes = list(gen)
    for idx, k in enumerate(tunes):
        #print(len(k))
        try: 
            song = art_songs[name][idx]
            instruments, bass_idxs, melody_idxs = get_instruments_from_NoteSequence(k)
            if len(instruments) > 0:
                all_instruments.add(instruments.values())
            if len(bass_idxs) > 0 and len(melody_idxs) > 0:
                filepaths.append(f'{name}/{song}')
                print(filepaths[-1])
        except Exception as e:
            print(e)
            continue

#%%
            
from utils.tools import write_list_as_csv_txt

write_list_as_csv_txt('bass_melody.txt', filepaths)


#%%

import shutil, os

dest_folder = '/home/sarantos/Documents/Music_AI/clean_midi_BM/'
for f in filepaths:
    file = dataset_dir + '/' + f + '.mid'
    shutil.copy(file, dest_folder)
    
#%%
    
dest_folder = '/home/sarantos/Documents/Music_AI/clean_midi_BM/'

midis = []
for root, subdirs, files in os.walk(dest_folder):
    for idx, f in enumerate(tqdm(files)):
        midis.append(midi_io.midi_file_to_note_sequence(dest_folder + '/' + f))
        midis[idx].filename = f
        #midis = [midi_io.midi_file_to_note_sequence(dest_folder + '/' + f) for f in tqdm(files)]
    
#%%
import copy

k = copy.deepcopy(midis[0])
instruments, bass_idxs, melody_idxs = get_instruments_from_NoteSequence(k)


#%%
import note_seq
from note_seq.protobuf import music_pb2

new_tune = note_seq.protobuf.music_pb2.NoteSequence()
new_tune.tempos.add(qpm=k.tempos[0].qpm)
new_tune.filename = k.filename
for idx, note in enumerate(k.notes):
    if note.instrument in bass_idxs + melody_idxs:
        new_tune.notes.add(pitch=note.pitch,
                           start_time=note.start_time,
                           end_time=note.end_time,
                           velocity=note.velocity)


note_seq.sequence_proto_to_midi_file(new_tune, new_tune.filename)
#%%
        
