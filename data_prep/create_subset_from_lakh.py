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
#CHANGE ME
clean_midi_dir = '/home/sarantos/Documents/Music_AI/clean_midi' #Full path of clean_midi dataset
dest_folder = '/home/sarantos/Documents/Music_AI/clean_midi_BM/' #Full path of clean_midi SUBSET(Tracks with only bass and melody)

#%%
'''
Remove files that cannot be transformed to a NoteSequence midi object
Only run it once
Should delete 276 files out of 17256 (Function should take around 30 minutes or so)
'''
del_files, n_files, errors = remove_files_from_clean_midi(clean_midi_dir)

#%%

art_midi, art_nfiles, art_songs = get_artists_midi_dict_generator(clean_midi_dir)


#%%

def get_midi_filepaths_with_bass_and_melody(art_midi, art_songs):
    """
    Parameters
    ----------
    art_midi : Dict with keys: artists name(string)
                         val: Generator of list of note_seq objects
                              (length of list == length of songs per artist)
    art_songs : Dict with keys: artist name (string)
                          val: List of song names

    Returns
    -------
    filepaths : relative filepaths of songs that have at least one channel 
                for bass and melody
    all_instruments: Set of all unique instruments

    """
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
    return filepaths, all_instruments

filepaths, all_instruments = get_midi_filepaths_with_bass_and_melody(art_midi, art_songs)


#%%


write_list_as_csv_txt('bass_melody.txt', filepaths)
text_file = open("bass_melody.txt", "r")
artists = text_file.read().split(',')
        
#%%

## Create new folder and copyfiles for clean_midi to dest_folder
for f in filepaths:
    file = clean_midi_dir + '/' + f + '.mid'
    shutil.copy(file, dest_folder)
    

#%%
from mido import MidiFile
read_midi = MidiFile
midis = read_midis_from_clean_midi_BM(folder=dest_folder, read_midi=read_midi)

#%%
import mido
def rewrite_mido_MidiFile_bass_melody(cv1, folder_name, bass_program=34, melody_program=54,
                                      delete_rest=True):
    mel_counter = 0
    bass_counter = 0
    tracks_remove = []
    for i, track in enumerate(cv1.tracks):
        print(i,track.name)
        if len(track.name) == 0:
            #print(i)
            for msg in track:
                #print(i, msg)
                #print(type(msg))
                if isinstance(msg, mido.messages.messages.Message):
                    print(i,'delete')
                    #print(track)
                    tracks_remove.append(track)
                    break
            #print('sound')
            continue
    
        if 'melody' in track.name.lower() and mel_counter == 0:
            mel_counter += 1
            #print(track.name)
            cv1.tracks[i].name = 'melody'
            for msg in cv1.tracks[i]:
                try:
                    msg.program = melody_program
                except:
                    pass
            for msg in cv1.tracks[i]:
                try:
                    msg.channel = 0
                except:
                    pass
            
            continue
        
        if 'bass' in track.name.lower() and bass_counter == 0:
            bass_counter += 1
            #print(track.name)
            cv1.tracks[i].name = 'bass'
            for msg in cv1.tracks[i]:
                try:
                    msg.program = bass_program
                except:
                    pass
            
            for msg in cv1.tracks[i]:
                try:
                    msg.channel = 1
                except:
                    pass
            continue
        
        tracks_remove.append(track)
        
    if delete_rest:
        for t in tracks_remove:
            cv1.tracks.remove(t)
    
    cv1.save(folder_name + '/' + cv1.filename)
    return cv1

folder_name = '/home/sarantos/Documents/Music_AI/CLEAN_MIDI_BM'

for i, midi in enumerate(tqdm(midis)):
    rewrite_mido_MidiFile_bass_melody(midi, folder_name, bass_program=34, melody_program=54,
                                      delete_rest=True)
    
#%%
    
midis = read_midis_from_clean_midi_BM(folder=folder_name, read_midi=midi_io.midi_file_to_note_sequence)
#%%

"""
def rewrite_mido_MidiFile_bass_melody(cv1, bass_program=34, melody_program=54, folder):
    mel_counter = 0
    bass_counter = 0
    tracks_remove = []
    for i, track in enumerate(cv1.tracks):
        #print(i,track.name)
        if len(track.name) == 0 and i==0:
            #for msg in track:
                #print(i, msg)
                #print('-')
            continue
        
        if 'melody' in track.name.lower() and mel_counter == 0:
            mel_counter += 1
            print(track.name)
            cv1.tracks[i].name = 'melody'
            for msg in cv1.tracks[i]:
                try:
                    msg.program = 54
                except:
                    pass
            for msg in cv1.tracks[i]:
                try:
                    msg.channel = 0
                except:
                    pass
            continue
        
        if 'bass' in track.name.lower() and bass_counter == 0:
            bass_counter += 1
            print(track.name)
            cv1.tracks[i].name = 'bass'
            for msg in cv1.tracks[i]:
                try:
                    msg.program = 34
                except:
                    pass
            for msg in cv1.tracks[i]:
                try:
                    msg.channel = 1
                except:
                    pass
            continue
        tracks_remove.append(track)
        
    for t in tracks_remove:
        cv1.tracks.remove(t)
    cv1.save(os.getcwd() + '/' + cv1.filename)

#%%
import copy
cv1 = copy.deepcopy(midis[17])
import mido

#cv1 = MidiFile('I Feel Love.1_gmajor_3_4.mid')

mel_counter = 0
bass_counter = 0
tracks_remove = []
for i, track in enumerate(cv1.tracks):
    #print(i,track.name)
    if len(track.name) == 0:
        print(i)
        for msg in track:
            #print(i, msg)
            #print(type(msg))
            if isinstance(msg, mido.messages.messages.Message):
                print(i,'delete')
                #print(track)
                tracks_remove.append(track)
                break
        print('sound')
        continue

    if 'melody' in track.name.lower() and mel_counter == 0:
        mel_counter += 1
        #print(track.name)
        cv1.tracks[i].name = 'melody'
        for msg in cv1.tracks[i]:
            try:
                msg.program = 54
            except:
                pass
        
        for msg in cv1.tracks[i]:
            try:
                msg.channel = 0
            except:
                pass
        continue
    
    if 'bass' in track.name.lower() and bass_counter == 0:
        bass_counter += 1
        #print(track.name)
        cv1.tracks[i].name = 'bass'
        for msg in cv1.tracks[i]:
            try:
                msg.program = 34
            except:
                pass
        
        for msg in cv1.tracks[i]:
            try:
                msg.channel = 1
            except:
                pass
        continue
    
    tracks_remove.append(track)
    
    #cv1.tracks.remove(track)
for t in tracks_remove:
    cv1.tracks.remove(t)

cv1.save(os.getcwd() + '/' + cv1.filename)


#%%


folder = dest_folder
read_midi = MidiFile
midis = []
for root, subdirs, files in os.walk(folder):
    for idx, f in enumerate(tqdm(files)):
        midis.append(read_midi(folder + '/' + f))
        midis[idx].filename = f
        if idx == 1:
            break
        #midis = [midi_io.midi_file_to_note_sequence(dest_folder + '/' + f) for f in tqdm(files)]


midis = read_midis_from_clean_midi_BM(folder=dest_folder)
import copy


#%%
k = copy.deepcopy(midis[0])

k = midi_io.midi_file_to_note_sequence(dest_folder + 'I Feel Love.1.mid')
instruments, bass_idxs, melody_idxs = get_instruments_from_NoteSequence(k)
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

"""