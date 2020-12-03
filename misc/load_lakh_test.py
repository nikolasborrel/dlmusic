import numpy as np
from music_utils.notes import get_midi_pitch_to_note_names_dict
from note_seq import midi_io
#from note_seq.protobuf import music_pb2
import pretty_midi as pm
from utils.tools import read_json
#import loaders.dataloader_midi as loader
from note_seq.protobuf.music_pb2 import NoteSequence
from loaders.dataloader_midi import remove_files_from_clean_midi, get_artists_midi_dict_generator

# https://github.com/magenta/magenta/blob/master/magenta/pipelines/note_sequence_pipelines_test.py
import matplotlib.pyplot as plt

#%%
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
    instruments, bass_idxs, melody_idxs = get_stats_from_NoteSequence(k)
    print(instruments.values())
    print('bass: ', bass_idxs)
    print('melody: ', melody_idxs) 