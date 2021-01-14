from collections import OrderedDict
import math
import copy
from music_utils.notes import midi_number_to_note
import matplotlib.pyplot as plt
import note_seq
from note_seq import events_lib
from note_seq import constants
from note_seq import melodies_lib
from note_seq import sequences_lib
from note_seq.sequences_lib import _extract_subsequences, concatenate_sequences, split_note_sequence_on_silence, extract_subsequence
from note_seq.melodies_lib import Melody
from note_seq import encoder_decoder
from note_seq import melody_encoder_decoder
from note_seq import midi_io
from note_seq.protobuf import music_pb2
from note_seq.constants import DEFAULT_QUARTERS_PER_MINUTE
from note_seq.protobuf.generator_pb2 import GeneratorOptions
from note_seq.protobuf.music_pb2 import NoteSequence
from typing import Tuple, List, Optional
from utils.tools import flatten
from collections import Counter
import numpy as np
import utils.paths as paths
from note_seq import midi_io
NO_KEY_SIGNATURE = 'no_key_signatures_found'
MORE_THAN_ONE_KEY_SIGNATURE = 'more_than_one_key_signature_found'
SKIPPED_DUE_TO_RANGE = 'skipped_due_to_range_exceeded'

class TokenizerMonophonic():

    DEFAULT_MIN_NOTE = 48
    DEFAULT_MAX_NOTE = 84
    DEFAULT_STEPS_PER_QUARTER = 4 #How many notes to encode in 1/4th of a bar - def  value of 4 --> 16th note quantization
    
    # min=60 max=72 -> one octave
    def __init__(self, split_in_bar_chunks=4, min_note=DEFAULT_MIN_NOTE, max_note=DEFAULT_MAX_NOTE, steps_per_quarter=DEFAULT_STEPS_PER_QUARTER) -> None:
        self._song_parts_lead = []
        self._song_parts_accomp = []
        self._min_note = min_note
        self._max_note = max_note
        self._split_in_bar_chunks = split_in_bar_chunks
        self._steps_per_quarter = steps_per_quarter
        self.stats = dict()
        self.midi_names = []
        self.splits_per_midi = []
        self.vocab = [-2,-1] + list(range(self._min_note, self._max_note))
        self.counter = 0

        self._encoder_decoder = encoder_decoder.OneHotEventSequenceEncoderDecoder(
            melody_encoder_decoder.MelodyOneHotEncoding(min_note, max_note))

        # Additional labels are NO_EVENT = 0 and NOTE_OFF = 1
        assert(self._encoder_decoder.input_size, max_note - min_note + 2)
        assert(self._encoder_decoder.num_classes, max_note - min_note + 2)

    def add_songs(self, seqs: List[NoteSequence], instruments: List[int], ignore_polyphonic_notes=True):
        '''
        sequences is a list of list of note_seq, each inner list corresponding to (part of) a song
        '''
        self.n_midi_read = len(seqs)
        
        if len(instruments) == 0:
            raise Exception("Instruments are empty")

        if len(instruments) > 2:
            raise Exception("Currently number of instruments to extract can only be one or two")

        for s in seqs:
            self.add_song(s, instruments, ignore_polyphonic_notes=ignore_polyphonic_notes)
        
        print(self.stats)

    def _truncate_to_bars(self, note_seq):
        bar_length_secs = self._calc_bar_length_sec(note_seq)
        num_bars_trunc = math.floor(note_seq.total_time / bar_length_secs)
        
        return sequences_lib.trim_note_sequence(note_seq, 0, num_bars_trunc*bar_length_secs)

    def add_song(self, seq_raw: NoteSequence, instruments: List[int], ignore_polyphonic_notes=True):
        '''
        sequences is a list of list of note_seq, each inner list corresponding to (part of) a song
        '''
        self.midi_names.append(seq_raw.filename)

        if len(seq_raw.tempos) != 1 or len(seq_raw.time_signatures) != 1:            
            #print("skipping song: multiple tempo or time signatures")
            return
        
        if seq_raw.time_signatures[0].numerator != 4 or seq_raw.time_signatures[0].denominator != 4:
            #print(f"skipping song: only 4/4 time signature supported, got {seq_raw.time_signatures[0].numerator}/{seq_raw.time_signatures[0].denominator}")
            return

        # seq_time_change_split = sequences_lib.split_note_sequence_on_time_changes(seq_transp_c)[0]

        split_num_bars = 1        

        seq_transp_c = self._transpose(seq_raw, 0)

        bar_length_secs = self._calc_bar_length_sec(seq_transp_c) * split_num_bars
        gap_secs = bar_length_secs * split_num_bars

        # TODO: check that the splitted sequences have not been shifted rythimically (e.g. offbeat)
        #       if melody doesn't start on downbeat
        seqs_split_silence = sequences_lib.split_note_sequence_on_silence(seq_transp_c, 
                                                                          instr=instruments[0], 
                                                                          remove_silence=True, 
                                                                          gap_seconds=gap_secs)

        if len(seqs_split_silence) == 0:
            # problematic data - skip
            #print("skipping song: no melody to split")
            return

        seqs_split_silence_trunc = list(map(self._truncate_to_bars, seqs_split_silence))
        seq_split_silence        = sequences_lib.concatenate_sequences(seqs_split_silence_trunc)
        seqs_split_bars          = self._split_on_bars(seq_split_silence)

        if len(seqs_split_bars) == 0:
            # TODO: we could eventually pad in the split to bars function
            print("skipping song: all melody lengths have lengths less than 1 bar")
            return

        bars_length_sec = self._calc_bar_length_sec(seqs_split_bars[-1]) * self._split_in_bar_chunks
        if seqs_split_bars[-1].total_time < bars_length_sec:
            seqs_split_bars = seqs_split_bars[0:-2]
        
        for seq in seqs_split_bars:                        
            quantized_sequence = sequences_lib.quantize_note_sequence(
                seq, steps_per_quarter=self._steps_per_quarter)

            # EXTRACT FIRST INSTRUMENT
            melody0 = Melody()
            melody0.from_quantized_sequence(
                quantized_sequence, instrument=instruments[0], 
                ignore_polyphonic_notes=ignore_polyphonic_notes, gap_bars=100000000)

            # squeeze midi into octaves determined by min_note and max_note
            # transpose_to_key = 0 -> not working properly!
            melody0.squash(self._min_note, self._max_note)
            melody1 = None
            
            if len(instruments) == 2:
                # EXTRACT SECOUND INSTRUMENT
                melody1 = Melody()
                melody1.from_quantized_sequence(
                    quantized_sequence, instrument=instruments[1], 
                    ignore_polyphonic_notes=ignore_polyphonic_notes, gap_bars=100000000)

                # squeeze midi into octaves determined by min_note and max_note
                # transpose_to_key = 0 -> not working properly!
                melody1.squash(self._min_note,self._max_note) 

                if len(melody0) > 0 and len(melody1) > 0:
                    #num_steps_truncated = math.ceil(len(melody0.steps) / melody0.steps_per_bar) * melody0.steps_per_bar
                    num_steps_truncated = self._split_in_bar_chunks * melody0.steps_per_bar
                    melody0.set_length(num_steps_truncated)
                    melody1.set_length(num_steps_truncated)

                    self.debug_encoding_length(melody0, num_steps_truncated-1)
                    self.debug_encoding_length(melody1, num_steps_truncated-1)

                    self._song_parts_lead.append(melody0)
                    self._song_parts_accomp.append(melody1)                    

            elif len(melody0) > 0:
                # num_steps_truncated = math.ceil(len(melody0.steps) / melody0.steps_per_bar) * melody0.steps_per_bar
                num_steps_truncated = self._split_in_bar_chunks * melody0.steps_per_bar
                melody0.set_length(num_steps_truncated)
                
                self._song_parts_lead.append(melody0)   
            
        self.splits_per_midi.append((len(self._song_parts_lead), \
                                     len(self._song_parts_accomp)))
        
        #nan = [midi_io.sequence_proto_to_midi_file(seq.to_sequence(), paths.midi_dir_one_song + f'mel_{self.counter}_{i}.mid') for i, seq in enumerate(self._song_parts_lead)]
        #nan = [midi_io.sequence_proto_to_midi_file(seq.to_sequence(), paths.midi_dir_one_song + f'bass_{self.counter}_{i}.mid') for i, seq in enumerate(self._song_parts_accomp)]
        self.counter += 1

    def write_dataset(self, path_out):
        [midi_io.sequence_proto_to_midi_file(seq.to_sequence(), path_out + f'mel_{i}.mid') for i, seq in enumerate(self._song_parts_lead)]
        [midi_io.sequence_proto_to_midi_file(seq.to_sequence(), path_out + f'bass_{i}.mid') for i, seq in enumerate(self._song_parts_accomp)]
    
    def debug_encoding_length(self, melody, expected):
        input_one_hot, _ = self._encoder_decoder.encode(melody)
        if len(input_one_hot) != expected:
            raise Exception(f"wrong length: expected {expected}, got {len(input_one_hot)}")

    def to_midi(self, outputs, path_out_dir, filename='out.mid'):
        """
        outputs: one-hot encoded tensor (seq_len, vocab_size)
        """
        events = []
        for event_one_hot in outputs:  #for each event
            label = np.argmax(event_one_hot)   #find index of maximum value from one-hot
            events.append(self._encoder_decoder.class_index_to_event(label, events))

        print(events)

        mel_pred = note_seq.Melody(events)
        seq_pred = mel_pred.to_sequence()

        path_out = path_out_dir + filename
        midi_io.sequence_proto_to_midi_file(seq_pred, path_out)
    
    def events_to_melody(events, path_out=None):
        seq = note_seq.Melody(events)
        if path_out is not None:
            midi_io.sequence_proto_to_midi_file(seq.to_sequence(), path_out)
        return seq

    def plot_melodies_with_vocab(self, mel_seq: Melody, bass_seq: Melody):
        assert len(mel_seq) == len(bass_seq)
        vocab = self.vocab
        vocab[0] = -2
        vocab[1] = -1

        bass = np.zeros((len(vocab), len(bass_seq)))
        mel = np.zeros((len(vocab), len(mel_seq)))
        vocab_to_idx = {key: val for key, val in zip(vocab, reversed(range(0, len(vocab))))}

        for i in range(len(mel_seq)):
            mel_note = mel_seq._events[i]
            bass_note = bass_seq._events[i]
            mel[vocab_to_idx[mel_note], i] = 1
            bass[vocab_to_idx[bass_note], i] = 1
        
        fig, axs = plt.subplots(nrows=1, ncols=2)
        axs[0].imshow(mel)
        axs[0].set_title('Melody')
        
        axs[1].imshow(bass)
        axs[1].set_title('Bass')

    def get_melody_histogram(self, melody: Melody):
        events = melody._events
        bincount = Counter(events)
        return sorted(bincount.items())

    def _transpose(self, note_seq, to_key, 
            min_pitch=constants.MIN_MIDI_PITCH, max_pitch=constants.MAX_MIDI_PITCH) -> NoteSequence:
        """Transposes a note sequence by the specified amount."""

        def update_stats(key):                 
            if key in self.stats:
                self.stats[key] += 1
            else:
                self.stats[key] = 1

        note_seq_key = 0 # C major

        if len(note_seq.key_signatures) == 0:            
            update_stats(NO_KEY_SIGNATURE)        
        elif len(note_seq.key_signatures) > 1:
            note_seq_key = note_seq.key_signatures[0].key
            update_stats(MORE_THAN_ONE_KEY_SIGNATURE)
        
        amount = to_key - note_seq_key
        
        return sequences_lib.transpose_note_sequence(note_seq, amount, min_allowed_pitch=min_pitch, max_allowed_pitch=max_pitch)[0]

    def _createMelodyObject(self, sequence: NoteSequence, pad_end=False) -> Melody:
        melody = Melody()
        quantized_sequence = sequences_lib.quantize_note_sequence(
            sequence, steps_per_quarter=self._steps_per_quarter)
        melody.from_quantized_sequence(quantized_sequence, pad_end=pad_end, gap_bars=10000000)
        return melody

    def _calc_bar_length(self, qpm, steps_per_bar):
        '''
        qpm: quarter per minute
        
        divide 60 sec by qpm to get quarter notes per sec; divide by steps per quarter to get time per step

        returns: the bar length in seconds
        '''
        seconds_per_step = 60.0 / qpm / self._steps_per_quarter
        return steps_per_bar*seconds_per_step

    def _calc_steps_per_bar(self, note_seq):
        quantized_sequence = sequences_lib.quantize_note_sequence(
            note_seq, steps_per_quarter=self._steps_per_quarter)

        steps_per_bar_float = sequences_lib.steps_per_bar_in_quantized_sequence(
            quantized_sequence)

        if steps_per_bar_float % 1 != 0:
            raise events_lib.NonIntegerStepsPerBarError(
                'There are %f timesteps per bar. Time signature: %d/%d' %
                (steps_per_bar_float, quantized_sequence.time_signatures[0].numerator,
                quantized_sequence.time_signatures[0].denominator))
        
        return int(steps_per_bar_float)

    def _calc_bar_length_sec(self, note_seq):
        
        if len(note_seq.tempos) != 1:
            raise Exception(f'Only one tempo indication should be present: {len(note_seq.tempos)} found')

        qpm = note_seq.tempos[0].qpm

        steps_per_bar = self._calc_steps_per_bar(note_seq)
        return self._calc_bar_length(qpm, steps_per_bar)

    def _split_on_bars(self, note_seq):

        bars_length_sec = self._calc_bar_length_sec(note_seq) * self._split_in_bar_chunks

        return sequences_lib.split_note_sequence(note_seq, bars_length_sec) # only works on non-quantized sequences (for some reason...)

    @property
    def song_parts_lead(self):
        return self._song_parts_lead

    @property
    def song_parts_accomp(self):
        return self._song_parts_accomp

    # The encoder/decoder for a monophonic sequence
    @property
    def encoder_decoder(self):
        return self._encoder_decoder

    # An integer count of the total number of documents that were used to fit the Tokenizer.
    @property
    def song_count(self):
        return len(self._song_parts_lead)

    @property
    def vocab_size(self):
        return self._encoder_decoder.num_classes

"""
  Melody events are integers in range [-2, 127] (inclusive), where negative
  values are the special event events: MELODY_NOTE_OFF, and MELODY_NO_EVENT.
  Non-negative values [0, 127] are note-on events for that midi pitch. A note
  starts at a non-negative value (that is the pitch), and is held through
  subsequent MELODY_NO_EVENT events until either another non-negative value is
  reached (even if the pitch is the same as the previous note), or a
  MELODY_NOTE_OFF event is reached. A MELODY_NOTE_OFF starts at least one step
  of silence, which continues through MELODY_NO_EVENT events until the next
  non-negative value.
  """