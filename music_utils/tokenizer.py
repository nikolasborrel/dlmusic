from collections import OrderedDict
import math
import copy
from music_utils.notes import midi_number_to_note
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
import numpy as np

NO_KEY_SIGNATURE = 'no_key_signatures_found'
MORE_THAN_ONE_KEY_SIGNATURE = 'more_than_one_key_signature_found'
SKIPPED_DUE_TO_RANGE = 'skipped_due_to_range_exceeded'

class TokenizerMonophonic():

    DEFAULT_MIN_NOTE = 48
    DEFAULT_MAX_NOTE = 84
    DEFAULT_STEPS_PER_QUARTER = 4
    
    # min=60 max=72 -> one octave
    def __init__(self, max_bars_chunk=32, min_note=DEFAULT_MIN_NOTE, max_note=DEFAULT_MAX_NOTE, steps_per_quarter=DEFAULT_STEPS_PER_QUARTER) -> None:
        self._song_parts_lead = []
        self._song_parts_accomp = []
        self._min_note = min_note
        self._max_note = max_note
        self._max_bars_chunk = max_bars_chunk
        self._steps_per_quarter = steps_per_quarter

        self._encoder_decoder = encoder_decoder.OneHotEventSequenceEncoderDecoder(
            melody_encoder_decoder.MelodyOneHotEncoding(min_note, max_note))

        # Additional labels are NO_EVENT = 0 and NOTE_OFF = 1
        assert(self._encoder_decoder.input_size, max_note - min_note + 2)
        assert(self._encoder_decoder.num_classes, max_note - min_note + 2)

    def _split_on_bars(self, note_seq):

        def calc_steps_per_bar():       
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

        if len(note_seq.tempos) != 1:
            raise Exception(f'Only one tempo indication should be present: {len(note_seq.tempos)} found')

        qpm = note_seq.tempos[0].qpm

        steps_per_bar = calc_steps_per_bar()
        bar_length_sec = calc_bar_length(qpm, self._steps_per_quarter, steps_per_bar) * self._max_bars_chunk

        return sequences_lib.split_note_sequence(note_seq, bar_length_sec) # only works on non-quantized sequences (for some reason...)

    def add_songs(self, seqs: List[NoteSequence], instruments: List[int], ignore_polyphonic_notes=True):
        '''
        sequences is a list of list of note_seq, each inner list corresponding to (part of) a song
        '''
        
        if len(instruments) == 0:
            raise Exception("Instruments are empty")

        if len(instruments) > 2:
            raise Exception("Currently number of instruments to extract can only be one or two")

        stats = {}

        seqs_transposed_c   = list(map(lambda ns: self._transpose(ns, 0, stats), seqs))        
        seqs_time_change_split = flatten(list(map(sequences_lib.split_note_sequence_on_time_changes, seqs_transposed_c)))        
        seqs_preprocessed = flatten(list(map(self._split_on_bars, seqs_time_change_split)))

        print(stats)

        self.loading_errors = []
        for seq in seqs_preprocessed:                        
            quantized_sequence = sequences_lib.quantize_note_sequence(
                seq, steps_per_quarter=self._steps_per_quarter)

            # EXTRACT FIRST INSTRUMENT
            melody0 = Melody()
            melody0.from_quantized_sequence(
                quantized_sequence, instrument=instruments[0], 
                ignore_polyphonic_notes=ignore_polyphonic_notes, gap_bars=100000000)

            # squeeze midi into octaves determined by min_note and max_note
            melody0.squash(
                self._min_note,
                self._max_note) # transpose_to_key = 0 -> not working properly!

            melody1 = None
            
            if len(instruments) == 2:
                # EXTRACT SECOUND INSTRUMENT
                melody1 = Melody()
                melody1.from_quantized_sequence(
                    quantized_sequence, instrument=instruments[1], 
                    ignore_polyphonic_notes=ignore_polyphonic_notes, gap_bars=100000000)

                # squeeze midi into octaves determined by min_note and max_note
                melody1.squash(
                    self._min_note,
                    self._max_note) # transpose_to_key = 0 -> not working properly!

                if len(melody0) > 0 and len(melody1) > 0:
                    lead_and_accomp = self._remove_silence_lead_accomp(melody0, melody1, gap_bars=1)
                    
                    if lead_and_accomp != None:
                        self._song_parts_lead.extend(lead_and_accomp[0])
                        self._song_parts_accomp.extend(lead_and_accomp[1])
            
            elif len(melody0) > 0:
                mel_cleaned = self._remove_silence(melody0, gap_bars=1)
                if mel_cleaned != None:
                    self._song_parts_lead.extend(mel_cleaned)
        
        if len(self.loading_errors) > 0:
            print("Not all midi files could not be used:")
            print(self.loading_errors)

    def to_midi(self, outputs, path_out_dir):
        events = []
        for output in outputs:
            label = np.argmax(output)
            events.append(self._encoder_decoder.class_index_to_event(label, events))

        print(events)

        mel_pred = note_seq.Melody(events)
        seq_pred = mel_pred.to_sequence()

        path_out = path_out_dir + 'out.mid'
        midi_io.sequence_proto_to_midi_file(seq_pred, path_out)

    def _remove_silence_lead_accomp(self, monophonic_lead: Melody, monophonic_accomp: Melody, gap_bars=1) -> Optional[Tuple[List[Melody],List[Melody]]]:

        """Split one lead sequence of notes of type Melody into many around gaps of silence 
            and splits accompanying music according to the lead.

        This function splits a NoteSequence into multiple NoteSequences, each of which
        contains no gaps of silence longer than `gap_seconds`. Each of the resulting
        NoteSequences is shifted such that the first note starts at time zero.

        Args:
            monophonic_lead: The lead Melody to split.
            monophonic_accomp: The accompanying Melody to split accordingly.
            gap_bars: The maximum amount of contiguous silence to allow within a
                NoteSequence, in num bars.

        Returns:
            A Python list of tuples of Melody.
        """
        
        qpm = 120
        note_seq_lead = monophonic_lead.to_sequence(qpm=qpm)
        note_seq_accomp = monophonic_accomp.to_sequence(qpm=qpm)

        if note_seq_accomp.total_time < note_seq_lead.total_time:
            # truncate melody to length of accompagment
            note_seq_lead = extract_subsequence(note_seq_lead, 0, note_seq_accomp.total_time)

        bar_length_secs = calc_bar_length(qpm, monophonic_lead.steps_per_quarter, monophonic_lead.steps_per_bar)
        gap_seconds = bar_length_secs * gap_bars
        
        note_seqs_lead_split = split_note_sequence_on_silence(note_seq_lead, gap_seconds)
        
        note_seqs_accomp_split = []
        note_seqs_lead_split_tmp = []

        for split in note_seqs_lead_split:
            if split.total_time > bar_length_secs: # we ignore splits smaller than 1 bar                
                start_time = split.subsequence_info.start_time_offset
                end_time = start_time + split.total_time

                ns_accomp_split = extract_subsequence(note_seq_accomp, start_time, end_time)
                
                if ns_accomp_split.total_time > bar_length_secs:
                    note_seqs_lead_split_tmp.append(split)
                    note_seqs_accomp_split.append(ns_accomp_split)

        if len(note_seqs_lead_split_tmp) == 0:
            return None

        note_seqs_lead_split = note_seqs_lead_split_tmp

        if len(note_seqs_lead_split) != len(note_seqs_accomp_split):
            raise Exception("Note sequences have different lenghts")
            
        melodies_lead_padded = []
        melodies_accomp_padded = []
        
        for i in range(len(note_seqs_lead_split)):
            mel_lead = self._createMelodyObject(note_seqs_lead_split[i])
            mel_accomp = self._createMelodyObject(note_seqs_accomp_split[i])
            
            num_steps_truncated = math.ceil(len(mel_lead.steps) / mel_lead.steps_per_bar) * mel_lead.steps_per_bar
            mel_lead.set_length(num_steps_truncated)
            mel_accomp.set_length(num_steps_truncated)

            melodies_lead_padded.append(mel_lead)
            melodies_accomp_padded.append(mel_accomp)
            
        return (melodies_lead_padded, melodies_accomp_padded)

    def _transpose(self, note_seq, to_key, stats, 
            min_pitch=constants.MIN_MIDI_PITCH, max_pitch=constants.MAX_MIDI_PITCH) -> NoteSequence:
        """Transposes a note sequence by the specified amount."""

        def update_stats(key):                 
            if key in stats:
                stats[key] += 1
            else:
                stats[key] = 1            

        note_seq_key = 0 # C major

        if len(note_seq.key_signatures) == 0:            
            update_stats(NO_KEY_SIGNATURE)        
        elif len(note_seq.key_signatures) > 1:
            note_seq_key = note_seq.key_signatures[0].key
            update_stats(MORE_THAN_ONE_KEY_SIGNATURE)
        
        amount = to_key - note_seq_key
        
        return sequences_lib.transpose_note_sequence(note_seq, amount, min_allowed_pitch=min_pitch, max_allowed_pitch=max_pitch)[0]

    def _remove_silence(self, monophonic_mel: Melody, gap_bars=1) -> Optional[List[Melody]]:

        """Split one lead sequence of notes of type Melody into many around gaps of silence

        Args:
            monophonic_mel: The Melody to split.
            gap_bars: The maximum amount of contiguous silence to allow within a
                NoteSequence, in num bars.

        Returns:
            A Python list of Melodys.
        """
        
        qpm = 120
        note_seq_lead = monophonic_mel.to_sequence(qpm=qpm)        

        bar_length_secs = calc_bar_length(qpm, monophonic_mel.steps_per_quarter, monophonic_mel.steps_per_bar)
        gap_seconds = bar_length_secs * gap_bars
        
        note_seqs_lead_split = split_note_sequence_on_silence(note_seq_lead, gap_seconds)        
        note_seqs_lead_split_tmp = []

        for split in note_seqs_lead_split:
            if split.total_time > bar_length_secs: # we ignore splits smaller than 1 bar
                note_seqs_lead_split_tmp.append(split)

        if len(note_seqs_lead_split_tmp) == 0:
            return None

        note_seqs_lead_split = note_seqs_lead_split_tmp

        return list(map(self._createMelodyObject, note_seqs_lead_split))

    def _transpose(self, note_seq, to_key, stats, 
            min_pitch=constants.MIN_MIDI_PITCH, max_pitch=constants.MAX_MIDI_PITCH) -> NoteSequence:
        """Transposes a note sequence by the specified amount."""

        def update_stats(key):                 
            if key in stats:
                stats[key] += 1
            else:
                stats[key] = 1            

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

def calc_bar_length(qpm, steps_per_quarter, steps_per_bar):
    '''
    qpm: quarter per minute
    
    divide 60 sec by qpm to get quarter notes per sec; divide by steps per quarter to get time per step

    returns: the bar length in seconds
    '''
    seconds_per_step = 60.0 / qpm / steps_per_quarter
    return steps_per_bar*seconds_per_step