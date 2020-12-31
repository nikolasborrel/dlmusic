from collections import OrderedDict
import math
import copy
from music_utils.notes import midi_number_to_note
import note_seq
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
from typing import Tuple, List


class TokenizerMonophonic():

    DEFAULT_MIN_NOTE = 48
    DEFAULT_MAX_NOTE = 84

    # min=60 max=72 -> one octave
    def __init__(self, min_note=DEFAULT_MIN_NOTE, max_note=DEFAULT_MAX_NOTE) -> None:
        # self._note_counts = OrderedDict()
        # self._note_pieces = {}
        # self._note_index = {}
        # self._index_note = {}
        self._songs = []
        self._song_count = 0
        self._min_note = min_note
        self._max_note = max_note

        self._encoder_decoder = encoder_decoder.OneHotEventSequenceEncoderDecoder(
            melody_encoder_decoder.MelodyOneHotEncoding(min_note, max_note))  # min_note=DEFAULT_MIN_NOTE, max_note=DEFAULT_MAX_NOTE

        # Additional labels are NO_EVENT = 0 and NOTE_OFF = 1
        assert(self._encoder_decoder.input_size, max_note - min_note + 2)
        assert(self._encoder_decoder.num_classes, max_note - min_note + 2)

    def add_songs_from_sequences(self, songs: List[NoteSequence], instruments: Tuple[int, int], steps_per_quarter=4, ignore_polyphonic_notes=True):
        '''
        sequences is a list of list of note_seq, each inner list corresponding to (part of) a song
        '''
        NoteSequence()
        self.loading_errors = []
        for sequence in songs:
            try:
                quantized_sequence = sequences_lib.quantize_note_sequence(
                    sequence, steps_per_quarter=steps_per_quarter)
            except Exception as e:
                self.loading_errors.append((sequence.filename, e))
                continue

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
                mel_bass_silence_removed = self._remove_silence(melody0, melody1, steps_per_quarter, gap_bars=1)

                self._song_count = self._song_count + len(mel_bass_silence_removed)
                self._songs.extend(mel_bass_silence_removed)
        
        if len(self.loading_errors) > 0:
            print("Not all midi files could not be used:")
            print(self.loading_errors)

    def _remove_silence(self, monophonic_lead: Melody, monophonic_accomp: Melody, steps_per_quarter, gap_bars=1) -> List[Tuple[Melody, Melody]]:

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
        note_seqs_accomp = monophonic_accomp.to_sequence(qpm=qpm)
        
        if note_seqs_accomp.total_time < note_seq_lead.total_time:
            # truncate melody to length of accompagment
            note_seq_lead = extract_subsequence(note_seq_lead, 0, note_seqs_accomp.total_time)

        seconds_per_step = 60.0 / qpm / monophonic_lead.steps_per_quarter
        bar_length_secs = monophonic_lead.steps_per_bar*seconds_per_step
        gap_seconds = bar_length_secs * gap_bars

        note_seqs_lead_split = split_note_sequence_on_silence(note_seq_lead, gap_seconds)        
        note_seqs_accomp_split = []

        note_seqs_lead_split_tmp = []

        for split in note_seqs_lead_split:
            if split.total_time < 1e-4:
                continue

            note_seqs_lead_split_tmp.append(split)

            start_time = split.subsequence_info.start_time_offset
            end_time = start_time + split.total_time

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
        
        return [truncate_to_bars(melodies_lead[i], melodies_accomp[i]) for i in range(len(melodies_lead))]

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
