import note_seq
from magenta.models.polyphony_rnn import polyphony_lib
from note_seq import encoder_decoder
from magenta.models.polyphony_rnn import polyphony_encoder_decoder
from note_seq import midi_io
from note_seq.protobuf import music_pb2
from note_seq.constants import DEFAULT_QUARTERS_PER_MINUTE
from note_seq.protobuf.generator_pb2 import GeneratorOptions
from note_seq.protobuf.music_pb2 import NoteSequence
from note_seq import sequences_lib # many useful utils for quantization, splitting sequences etc

"""
see polyphony_sequence_generator.py (in magenta.models.polyphone_rnn)
"""

input_file = '/Users/nikolasborrel/github/midi_data_out/melodies/piano_poly.mid'
out_file = '/Users/nikolasborrel/github/midi_data_out/melodies/piano_poly_out.mid'
out_file_trans = '/Users/nikolasborrel/github/midi_data_out/melodies/piano_poly_trans_out.mid'
out_file_pred = '/Users/nikolasborrel/github/midi_data_out/melodies/piano_poly_pred_out.mid'

min_note = 60
max_note = 72
transpose_to_key = 2

steps_per_quarter = 4 # default, resulting in 16th note quantization

note_seq_raw = midi_io.midi_file_to_note_sequence(input_file)
note_seq_quan = note_seq.quantize_note_sequence(note_seq_raw, steps_per_quarter)
extracted_seqs, stats = polyphony_lib.extract_polyphonic_sequences(note_seq_quan)

assert(len(extracted_seqs <= 1)) # docs states that only one poly list are extracted
poly_seq = extracted_seqs[0]

print(poly_seq)

seq1 = poly_seq.to_sequence() #qpm=60.0
midi_io.sequence_proto_to_midi_file(seq1, out_file)

poly_encoder = encoder_decoder.OneHotEventSequenceEncoderDecoder(
    polyphony_encoder_decoder.PolyphonyOneHotEncoding())

if len(note_seq_raw.key_signatures) > 1:
    print("WARNING: more than one key signatures were found - only the first signature is used.")
original_key = note_seq_raw.key_signatures[0].key
transpose_interval = transpose_to_key - original_key

# PolyphonicSequence doesn't have a transpose function (like Music for monohonic)
for e in poly_seq:
    if e.pitch != None:
        e.pitch = e.pitch + transpose_interval

seq1_trans = poly_seq.to_sequence() #qpm=60.0
midi_io.sequence_proto_to_midi_file(seq1_trans, out_file_trans)

inputs, labels = poly_encoder.encode(poly_seq)
print(inputs)
print(labels)

poly_pred = polyphony_lib.PolyphonicSequence(steps_per_quarter=steps_per_quarter)
events = []
for label in labels:
    event = poly_encoder.class_index_to_event(label, None)
    events.append(event)
    poly_pred.append(event)

print(events)

seq_pred = poly_pred.to_sequence()
midi_io.sequence_proto_to_midi_file(seq_pred, out_file_pred)