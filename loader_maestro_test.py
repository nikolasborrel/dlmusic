from loaders.dataloader_midi import load_midi_to_seq
import note_seq

sequences = load_midi_to_seq('../maestro-v2.0.0_small', recursive=True)

print(sequences.size)

seq1 = sequences[0]
seq2 = sequences[1]

seq1_extracted = note_seq.extract_subsequence(seq1, 0.0, 0.5)
seq2_extracted = note_seq.extract_subsequence(seq2, 0.0, 0.5)
print('------SEQ 1--------')
print(seq1_extracted)
print('------SEQ 2--------')
print(seq2_extracted)