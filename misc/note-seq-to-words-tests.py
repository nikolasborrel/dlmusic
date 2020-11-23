from collections import Counter
from note_seq import midi_io
from utils.tools import equal_temperament_scale
import numpy as np

def parse_note_seq_object(midi_note_seq):
    """
    Parameters
    ----------
    midi_note_seq : note_seq midi object
                    type -- note_seq.protobuf.music_pb2.NoteSequence

    Returns
    -------
    pitches : numpy 1D array of all pitch numbers
    velocities : numpy 1D array of all velocities
    start_times : numpy 1D array of all start_times
    end_times : numpy 1D array of all end_times

    """
    t_end = midi_note_seq.total_time
    pitches = np.array([])
    velocities = np.array([])
    start_times = np.array([])
    end_times = np.array([])
    for note in k.notes:
        pitches = np.append(pitches,note.pitch)
        velocities = np.append(velocities, note.velocity)
        start_times = np.append(start_times, note.start_time)
        end_times = np.append(end_times, note.end_time)
    
    return pitches, velocities, start_times, end_times



def ceil_midi_times(start_times, end_times, precision=3):
    """
    ceils to precision start_times and end_times
    """
    def ceil_to_precision(val, precision=precision):
        return np.round(val + 0.5 * 10**(-precision), precision)
    start_times = np.vectorize(ceil_to_precision)(start_times)
    end_times = np.vectorize(ceil_to_precision)(end_times)
    return start_times, end_times

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


def get_midi_pitch_to_note_names_dict():
    note_names, _ = equal_temperament_scale()
    for idx, note in enumerate(note_names):
        if note == 'A0':
            break
    midi_number_to_note = {number: note for number, note in zip(range(21,120), note_names[idx:])}
    return midi_number_to_note


#%%
#CHANGE ME BELOW
maestro_dir = '/home/sarantos/Documents/Music_AI/maestro-v2.0.0-midi/maestro-v2.0.0/'
filename = '2018/MIDI-Unprocessed_Chamber1_MID--AUDIO_07_R3_2018_wav--2.midi'

#%%
midi_path = maestro_dir + filename
k = midi_io.midi_file_to_note_sequence(midi_path)


pitches, velocities, start_times, end_times = parse_note_seq_object(k)


#Ceil start and end times to precision 3 
s_t, e_t = ceil_midi_times(start_times, end_times)
durations = e_t - s_t 

midi_number_to_note_dict = get_midi_pitch_to_note_names_dict()

#Get vocabulary spanning from 'A0' to 'B8'
vocab = list(midi_number_to_note_dict.values())
vocab_size = len(vocab)

words = [midi_number_to_note_dict[pitch] for pitch in pitches]
print(words)
