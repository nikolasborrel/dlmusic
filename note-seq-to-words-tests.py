import numpy as np
from music_utils.notes import get_midi_pitch_to_note_names_dict
from note_seq import midi_io

def ceil_to(val, precision=3):
    return np.round(val + 0.5 * 10**(-precision), precision)

#%%
#CHANGE ME BELOW
maestro_dir = '/Users/nikolasborrel/github/maestro-v2.0.0/'
filename = '2018/MIDI-Unprocessed_Chamber1_MID--AUDIO_07_R3_2018_wav--2.midi'

#%%
midi_path = maestro_dir + filename
k = midi_io.midi_file_to_note_sequence(midi_path)

note_seq_note_props = list( 
    map(lambda x: (x.pitch, x.velocity, x.start_time, x.end_time), k.notes)
    )

durations = list(map(lambda t: ceil_to(t[3]-t[2]), note_seq_note_props))

midi_number_to_note_dict = get_midi_pitch_to_note_names_dict()

#Get vocabulary spanning from 'A0' to 'B8'
vocab = list(midi_number_to_note_dict.values())
vocab_size = len(vocab)

words = [midi_number_to_note_dict[note[0]] for note in note_seq_note_props]

print(words[0:20])
print(durations[0:20])