from collections import OrderedDict
from music_utils.notes import midi_number_to_note

class MusicTokenizer():
    def __init__(self) -> None :
        self._note_counts = OrderedDict() 
        self._note_pieces = {} 
        self._note_index = {}
        self._index_note = {}
        self._piece_count = 0

    def fit_on_notes_pieces(self, notes_pieces):
        self._piece_count = len(notes_pieces)        

        unique_index_count = 0
        for piece_indx in range(len(notes_pieces)):
            piece_was_added = False

            note_seq = notes_pieces[piece_indx]
            for note in note_seq:                
                note_pitch_encoded = midi_number_to_note(note.pitch)           

                if note_pitch_encoded in self._note_counts:
                    self._note_counts[note_pitch_encoded] += 1
                else:
                    self._note_counts[note_pitch_encoded] = 1

                if note_pitch_encoded not in self._note_index:
                    self._note_index[note_pitch_encoded] = unique_index_count
                    self._index_note[unique_index_count] = note_pitch_encoded
                    unique_index_count += 1

                if not piece_was_added:
                    if note_pitch_encoded in self._note_pieces:
                        self._note_pieces[note_pitch_encoded] += 1
                    else:
                        self._note_pieces[note_pitch_encoded] = 1                    
                    piece_was_added = True
        
        unique_index_count = len(self._note_index)
        self._note_index['<UNK>'] = unique_index_count
        self._index_note[unique_index_count] = '<UNK>'

    def note_seqs_to_sequences(self, notes_pieces) -> [[]]:
        encoded_seqs = [[] for row in range(len(notes_pieces))]

        for piece_indx in range(len(notes_pieces)):
            notes = notes_pieces[piece_indx]

            for note_indx in range(len(notes)):
                note = notes[note_indx]                
                encoded_seqs[piece_indx].append(midi_number_to_note(note.pitch))
        
        return encoded_seqs

    # A (ordered) dictionary of words and their counts.
    @property
    def note_counts(self):
        return self._note_counts

    # A dictionary of words and how many documents each appeared in.
    @property
    def note_pieces(self):
        return self._note_pieces

    # A dictionary of notes and their uniquely assigned integers.
    @property
    def note_index(self):
        return self._note_index

    @property
    def index_note(self):
        return self._index_note

    # An integer count of the total number of documents that were used to fit the Tokenizer.
    @property
    def piece_count(self):
        return self._piece_count

    @property
    def vocab_size(self):
        return len(self._note_index)