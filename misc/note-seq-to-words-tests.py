import numpy as np
from music_utils.notes import get_midi_pitch_to_note_names_dict
from note_seq import midi_io
#from note_seq.protobuf import music_pb2
import pretty_midi as pm
from magenta.pipelines.note_sequence_pipelines import TranspositionPipeline
import loaders.dataloader_midi as loader

# https://github.com/magenta/magenta/blob/master/magenta/pipelines/note_sequence_pipelines_test.py

def ceil_to(val, precision=3):
    return np.round(val + 0.5 * 10**(-precision), precision)

#%%
#CHANGE ME BELOW
maestro_dir = '/Users/nikolasborrel/github/maestro-v2.0.0_small/'
filename = 'unnormal_pen.mid'


#%%
midi_path = maestro_dir + filename
k = midi_io.midi_file_to_note_sequence(midi_path)
k_trans = loader.transpose_note_seqs_to_c([k])[0]

print(f"Notes original pitch: {k.key_signatures[0]} \n{k.notes[0]}\n\n{k.notes[1]}")
print(f"Notes transposed pitch: {k_trans.key_signatures[0]} \n{k_trans.notes[0]}\n\n{k_trans.notes[1]}") # key=[empty] means key=0 => C

note_seq_note_props = list( 
    map(lambda x: (x.pitch, x.velocity, x.start_time, x.end_time, ceil_to(x.end_time-x.start_time)), k.notes)
    )

durations = note_seq_note_props[5]
midi_number_to_note_dict = get_midi_pitch_to_note_names_dict()
vocab = list(midi_number_to_note_dict.values())
vocab_size = len(vocab)
words = [midi_number_to_note_dict[note[0]] for note in note_seq_note_props]

print(words[0:20])
print(durations[0:20])