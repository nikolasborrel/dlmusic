from collections import OrderedDict
import math
import copy
from music_utils.notes import midi_number_to_note
import note_seq
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

flatten = lambda t: [item for sublist in t for item in sublist]

NO_KEY_SIGNATURE = 'no_key_signatures_found'
MORE_THAN_ONE_KEY_SIGNATURE = 'more_than_one_key_signature_found'
SKIPPED_DUE_TO_RANGE = 'skipped_due_to_range_exceeded'

class TokenizerMonophonic():

    DEFAULT_MIN_NOTE = 48
    DEFAULT_MAX_NOTE = 84

    # TODO: max_bars_chunk disabled
    
    # min=60 max=72 -> one octave
    def __init__(self, max_bars_chunk=32, min_note=DEFAULT_MIN_NOTE, max_note=DEFAULT_MAX_NOTE) -> None:
        # self._note_counts = OrderedDict()
        # self._note_pieces = {}
        # self._note_index = {}
        # self._index_note = {}
        self._songs = []
        self._song_count = 0
        self._min_note = min_note
        self._max_note = max_note
        self._max_bars_chunk = max_bars_chunk

        self._encoder_decoder = encoder_decoder.OneHotEventSequenceEncoderDecoder(
            melody_encoder_decoder.MelodyOneHotEncoding(min_note, max_note))  # min_note=DEFAULT_MIN_NOTE, max_note=DEFAULT_MAX_NOTE

        # Additional labels are NO_EVENT = 0 and NOTE_OFF = 1
        assert(self._encoder_decoder.input_size, max_note - min_note + 2)
        assert(self._encoder_decoder.num_classes, max_note - min_note + 2)

    def add_songs(self, seqs: List[NoteSequence], instruments:Tuple[int,int], steps_per_quarter=4, ignore_polyphonic_notes=True):
        '''
        sequences is a list of list of note_seq, each inner list corresponding to (part of) a song
        '''
        
        stats = {}

        seqs_transposed_c   = list(map(lambda ns: self._transpose(ns, 0, stats), seqs))        
        seqs_preprocessed = flatten(list(map(sequences_lib.split_note_sequence_on_time_changes, seqs_transposed_c)))
        
        print(stats)

        self.loading_errors = []
        for seq in seqs_preprocessed:        
            quantized_sequence = sequences_lib.quantize_note_sequence(
                seq, steps_per_quarter=steps_per_quarter)

            # EXTRACT FIRST INSTRUMENT
            melody0 = Melody()
            melody0.from_quantized_sequence(
                quantized_sequence, instrument=instruments[0], ignore_polyphonic_notes=ignore_polyphonic_notes, gap_bars=100000000)

            # squeeze midi into octaves determined by min_note and max_note and transposes to key = 0 => C major / A minor
            melody0.squash(
                self._min_note,
                self._max_note) # transpose_to_key = 0 -> not working properly!

            # EXTRACT SECOUND INSTRUMENT
            melody1 = Melody()
            melody1.from_quantized_sequence(
                quantized_sequence, instrument=instruments[1], ignore_polyphonic_notes=ignore_polyphonic_notes, gap_bars=100000000)

            # squeeze midi into octaves determined by min_note and max_note and transposes to key = 0 => C major / A minor
            melody1.squash(
                self._min_note,
                self._max_note) # transpose_to_key = 0 -> not working properly!

            if len(melody0) > 0 and len(melody1) > 0:
                silence_removed_tuple = self._remove_silence(melody0, melody1, steps_per_quarter, gap_bars=1)                

                if silence_removed_tuple != None:
                    self._song_count = self._song_count + len(silence_removed_tuple)
                    self._songs.extend(silence_removed_tuple)
        
        if len(self.loading_errors) > 0:
            print("Not all midi files could not be used:")
            print(self.loading_errors)

    def _remove_silence(self, monophonic_lead: Melody, monophonic_accomp: Melody, steps_per_quarter, gap_bars=1) -> Optional[List[Tuple[Melody, Melody]]]:

        """Split one lead sequence of notes of type Melody into many around gaps of silence 
            and splits accompagning music according to the lead.

        This function splits a NoteSequence into multiple NoteSequences, each of which
        contains no gaps of silence longer than `gap_seconds`. Each of the resulting
        NoteSequences is shifted such that the first note starts at time zero.

        Args:
            monophonic_lead: The lead Melody to split.
            monophonic_accomp: The accompagning Melody to split accordingly.
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

        seconds_per_step = 60.0 / qpm / monophonic_lead.steps_per_quarter
        bar_length_secs = monophonic_lead.steps_per_bar*seconds_per_step
        gap_seconds = bar_length_secs * gap_bars

        #note_seqs_lead = sequences_lib.split_note_sequence(note_seq_lead, self._max_bars_chunk)
        #note_seqs_lead_split = flatten(list(map(lambda ns: split_note_sequence_on_silence(ns, gap_seconds), note_seqs_lead)))
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

        def createMelodyObject(sequence: NoteSequence) -> Melody:
            melody = Melody()
            quantized_sequence = sequences_lib.quantize_note_sequence(
                sequence, steps_per_quarter=steps_per_quarter)
            melody.from_quantized_sequence(quantized_sequence, gap_bars=10000000)
            return melody

        def truncate_to_bars(mel_lead: Melody, mel_accomp: Melody) -> Tuple[Melody, Melody]:
            '''
            Truncate to last bar with notes filling the whole bar (we could also pad, but would 
            then add silence of no value)
            '''

            num_steps_truncated = math.floor(len(mel_lead.steps) / mel_lead.steps_per_bar) * mel_lead.steps_per_bar
            if num_steps_truncated == 0:
                # at least one bar is needed!
                num_steps_truncated = mel_lead.steps_per_bar

            mel_lead.set_length(num_steps_truncated)
            mel_accomp.set_length(num_steps_truncated)

            return (mel_lead, mel_accomp)

        melodies_lead = list(map(createMelodyObject, note_seqs_lead_split))
        melodies_accomp = list(map(createMelodyObject, note_seqs_accomp_split))
        
        if len(melodies_lead) != len(melodies_accomp):
            stop = True

        return [truncate_to_bars(melodies_lead[i], melodies_accomp[i]) for i in range(len(melodies_lead))]

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

    @property
    def songs(self):
        return self._songs

    # The encoder/decoder for a monophonic sequence
    @property
    def encoder_decoder(self):
        return self._encoder_decoder

    # # A (ordered) dictionary of words and their counts.
    # @property
    # def note_counts(self):
    #     return self._note_counts

    # # A dictionary of words and how many documents each appeared in.
    # @property
    # def note_pieces(self):
    #     return self._note_pieces

    # # A dictionary of notes and their uniquely assigned integers.
    # @property
    # def note_index(self):
    #     return self._note_index

    # @property
    # def index_note(self):
    #     return self._index_note

    # An integer count of the total number of documents that were used to fit the Tokenizer.
    @property
    def song_count(self):
        return self._song_count

    @property
    def vocab_size(self):
        return self._encoder_decoder.num_classes
