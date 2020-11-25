# inspired by 
# - https://machinelearningmastery.com/prepare-text-data-deep-learning-keras/
# - https://www.kdnuggets.com/2020/03/tensorflow-keras-tokenization-text-data-prep.html
from note_seq import midi_io
from music_utils.tokenizer import MusicTokenizer
from utils.hot_encoder import one_hot_encode_sequence
from utils.tools import split_list

#%%
#PARAMS
maestro_dir = '/Users/nikolasborrel/github/maestro-v2.0.0/'
filename = '2018/MIDI-Unprocessed_Chamber1_MID--AUDIO_07_R3_2018_wav--2.midi'
midi_path = maestro_dir + filename

one_piece_data = midi_io.midi_file_to_note_sequence(midi_path)

notes_pieces = split_list(one_piece_data.notes,4,3)

t = MusicTokenizer()

t.fit_on_notes_pieces(notes_pieces)

# summarize what was learned
print(t.note_counts)
print(t.piece_count)
print(t.note_index)
print(t.note_pieces)
# integer encode documents - TODO
#encoded_docs = t.texts_to_matrix(docs, mode='count')
#print(encoded_docs)

maxlen = 100

sequences = t.note_seqs_to_sequences(notes_pieces)
#test_padded = pad_sequences(sequences, maxlen=maxlen)

print("Testing sequences:\n", sequences)
#print("\nPadded testing sequences:\n", test_padded)
#print("\nPadded testing shape:",test_padded.shape)

vocab_size = len(t.note_counts)
one_hot = one_hot_encode_sequence(sequences[0], vocab_size, t.note_index)

print("One hot for piece with index 0:\n", one_hot)