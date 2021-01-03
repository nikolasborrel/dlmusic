# inspired by 
# - https://machinelearningmastery.com/prepare-text-data-deep-learning-keras/
# - https://www.kdnuggets.com/2020/03/tensorflow-keras-tokenization-text-data-prep.html
import sys
sys.path.append('../note_seq') # needed unless installing forked lib from github

from note_seq import midi_io
from music_utils.tokenizer import TokenizerMonophonic
from loaders.dataloader_midi import load_midi_to_seq

#PARAMS
midi_dir = '/Users/nikolasborrel/github/dlmusic_data/midi_data_out/tests/'
midi_dir_out = '/Users/nikolasborrel/github/dlmusic_data/midi_data_out/splitted/'

instruments = [0]
lead_instrument = ('melody', instruments[0])
name_instrument_map = { lead_instrument[0]  :  lead_instrument[1] }

print("Create...")
sequences = load_midi_to_seq(midi_dir, name_instrument_map, recursive=False)

if len(sequences) == 0:
    raise Exception(f'No midi files loaded')

print("Tokenize...")
t = TokenizerMonophonic(max_bars_chunk=8, min_note=60, max_note=72)
t.add_songs(sequences, instruments)

print("write to disk...")

for i, mel in enumerate(t.song_parts_lead):
    inputs_one_hot, label_not_used = t.encoder_decoder.encode(mel)  # MELODY

    print(f'#encodings melody: {len(inputs_one_hot)}')

    path_out_mel_test = f'{midi_dir_out}mel_split_silence_{i}.mid'
    path_out_bass_test = f'{midi_dir_out}bass_split_silence_{i}.mid'
    midi_io.sequence_proto_to_midi_file(mel.to_sequence(), path_out_mel_test)
