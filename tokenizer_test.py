# inspired by 
# - https://machinelearningmastery.com/prepare-text-data-deep-learning-keras/
# - https://www.kdnuggets.com/2020/03/tensorflow-keras-tokenization-text-data-prep.html
from note_seq import midi_io
from music_utils.tokenizer import MusicTokenizer
from utils.hot_encoder import one_hot_encode_sequence_dict

def pad_sequences(sequences: [[]], maxlen) -> [[]]:
    seqs_padded = [[0 for col in range(maxlen)] for row in range(len(sequences))]
    for i in range(0, len(sequences)-1):
        if len(sequences) < maxlen:
            seqs_padded[i,0:len(sequences)-1] = sequences
        else:
            seqs_padded[i,:] = sequences[0:maxlen-1]

    return seqs_padded

def split_note_seq(list_, chunk_size, overlap):
    if chunk_size < 1:
        raise Exception("chunk size too small")
    if overlap >= chunk_size:
        raise Exception("overlap too large")

    return [list_[i:i+chunk_size] for i in range(0, len(list_), chunk_size-overlap)]


#%%
#PARAMS
maestro_dir = '/Users/nikolasborrel/github/maestro-v2.0.0/'
filename = '2018/MIDI-Unprocessed_Chamber1_MID--AUDIO_07_R3_2018_wav--2.midi'
midi_path = maestro_dir + filename

test_data = [midi_io.midi_file_to_note_sequence(midi_path)]

t = MusicTokenizer()

t.fit_on_note_seqs(test_data)

# summarize what was learned
print(t.note_counts)
print(t.piece_count)
print(t.note_index)
print(t.note_pieces)
# integer encode documents - TODO
#encoded_docs = t.texts_to_matrix(docs, mode='count')
#print(encoded_docs)

maxlen = 100

test_sequences = t.note_seqs_to_sequences(test_data)
#test_padded = pad_sequences(test_sequences, maxlen=maxlen)

print("Testing sequences:\n", test_sequences)
#print("\nPadded testing sequences:\n", test_padded)
#print("\nPadded testing shape:",test_padded.shape)

vocab_size = len(t.note_counts)
one_hot = one_hot_encode_sequence_dict(t.note_counts, vocab_size, t.note_index)

print("One hot:\n", one_hot)