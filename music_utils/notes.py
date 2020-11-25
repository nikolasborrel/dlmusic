from collections import Counter
import numpy as np
from enum import Enum

class Notes(Enum):
    C = 'C'
    Cs = 'C#'
    D = 'D'
    Ds = 'D#'
    E = 'E'
    F = 'F'
    Fs = 'F#'
    G = 'G'
    Gs = 'G#'
    A = 'A'
    As = 'A#'
    B = 'B'

def equal_temperament_scale(n_octaves=9):    
    notes = [note.value for note in Notes] 
    notes_pitches = []
    for n in range(0,n_octaves):
        octave = [note + str(n) for note in notes]
        notes_pitches += octave
    f_ref = 440 #A4
    notes_freq = []
    for i in range(0, len(notes_pitches)):
        note_freq = f_ref * 2 ** ((- notes_pitches.index('A4') + i) / 12)
        notes_freq.append(note_freq)
    return notes_pitches, notes_freq

def get_midi_pitch_to_note_names_dict():
    note_names, _ = equal_temperament_scale()
    for idx, note in enumerate(note_names):
        if note == 'A0':
            break
    midi_number_to_note = {number: note for number, note in zip(range(21,120), note_names[idx:])}
    return midi_number_to_note

midi_pitch_to_note_names_dict = get_midi_pitch_to_note_names_dict() # static

def midi_number_to_note(midi_number):
    return midi_pitch_to_note_names_dict[midi_number]