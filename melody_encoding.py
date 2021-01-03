import note_seq
from note_seq import melodies_lib
from note_seq import encoder_decoder
from note_seq import melody_encoder_decoder
from note_seq import midi_io
from note_seq.protobuf import music_pb2
from note_seq.constants import DEFAULT_QUARTERS_PER_MINUTE
from note_seq.protobuf.generator_pb2 import GeneratorOptions
from note_seq.protobuf.music_pb2 import NoteSequence

input_file1 = '/Users/nikolasborrel/github/dlmusic_data/midi_data_out/melodies/mel1.mid'
input_file2 = '/Users/nikolasborrel/github/dlmusic_data/midi_data_out/melodies/mel2.mid'
out_file1 = '/Users/nikolasborrel/github/dlmusic_data/midi_data_out/melodies/mel1_out.mid'
out_file2 = '/Users/nikolasborrel/github/dlmusic_data/midi_data_out/melodies/mel2_out.mid'
out_file1_trans = '/Users/nikolasborrel/github/dlmusic_data/midi_data_out/melodies/mel1_trans_out.mid'
out_file1_pred = '/Users/nikolasborrel/github/dlmusic_data/midi_data_out/melodies/mel1_pred_out.mid'

# FOR IDEAS OF USING THE OUTPUT (hot encodings) DATA FROM THIS, SEE EVENTUALLY magenta.models.shared.eventss_rnn_model.py

melody1 = melodies_lib.midi_file_to_melody(input_file1)
seq = melody1.to_sequence()
midi_io.sequence_proto_to_midi_file(seq, out_file1)

min_note = 60
max_note = 72
transpose_to_key = 2
mel_encoder = encoder_decoder.OneHotEventSequenceEncoderDecoder(
    melody_encoder_decoder.MelodyOneHotEncoding(min_note, max_note)) # min_note=DEFAULT_MIN_NOTE, max_note=DEFAULT_MAX_NOTE

# Additional labels are NO_EVENT = 0 and NOTE_OFF = 1
assert(mel_encoder.input_size, max_note - min_note + 2) 
assert(mel_encoder.num_classes, max_note - min_note + 2)

# squeeze midi into octaves determined by min_note and max_note and transposes to key = 0 => C major / A minor
melody1.squash(
    min_note,
    max_note,
    transpose_to_key)

inputs, labels = mel_encoder.encode(melody1)
print(inputs)
print(labels)

# OR, if using batches (NOT SURE HOW TO USE / IF NEEDED - no labels? See magenta.models.shared.events_rnn_model.py)
melody2 = melodies_lib.midi_file_to_melody(input_file2)
melody2.squash(
    min_note,
    max_note,
    transpose_to_key)
    
inputs_batch = mel_encoder.get_inputs_batch([melody1, melody2], full_length=True)

events = []
for label in labels:
    events.append(mel_encoder.class_index_to_event(label, events))

print(events)

mel_pred = note_seq.Melody(events)
seq_pred = mel_pred.to_sequence()
midi_io.sequence_proto_to_midi_file(seq_pred, out_file1_pred)



# events = [100, 100, 107, 111, NO_EVENT, 99, 112, NOTE_OFF, NO_EVENT]
# melody = melodies_lib.Melody(events)
# melody.squash(
#     self.min_note,
#     self.max_note,
#     self.transpose_to_key)
# inputs, labels = self.med.encode(melody)
# expected_inputs = [
#     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
#     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
#     [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
#     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
# expected_labels = [2, 9, 13, 0, 13, 2, 1, 0]
# self.assertEqual(inputs, expected_inputs)
# self.assertEqual(labels, expected_labels)