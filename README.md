# dlmusic
Exploring the use of Deep Generative Models for live musical performance

## Pre-process data
[see also chapter 6 in the book "Music generation with Magenta"]

Dependecies: 
```
> pip install visual_midi
> pip install tables
```
For serializing data (not necessary at the moment)
```
> pip install magenta
```

Function for pre-processing data is found inside the directory `data_prep/`

The following [LMD](https://colinraffel.com/projects/lmd) dataset is used
* LMD-matched
* LMD-matched metadata
* Match scores

To get statistics:

Artist
```
python data_prep/01_artists_stats.py --sample_size=500 --path_dataset_dir="../midi_data/lmd_matched" --path_match_score="../midi_data/lmd_matched/match_scores.json"
```

Genre
```
python data_prep/02_genres_stats.py --sample_size=500 --path_dataset_dir="../midi_data/lmd_matched" --path_match_score="../midi_data/lmd_matched/match_scores.json" --last_fm_api_key="api_key"
```

Instrument
```
python data_prep/04_instrument_stats.py --sample_size=500 --path_dataset_dir="../midi_data/lmd_matched" --path_match_score="../midi_data/lmd_matched/match_scores.json"
```

Extract parts as midi, e,g. piano for jazz and blues genre, to output directory `path_output_dir`:
``` 
python data_prep/08_channel_extract_piano_tag.py --sample_size=500 --path_dataset_dir="../midi_data/lmd_matched" --path_match_score="../midi_data/lmd_matched/match_scores.json" --path_output_dir="../midi_data_out/piano_filtered" --last_fm_api_key="api_key" --tags="['jazz', 'blues']"
```

It is also possible to to write the extracted midi data as serialized NoteSequence data as Tensorflow `-tf` file for efficient storage and loading (can stil be used in e.g. pyTorch after loading - but let's wait with this and use our current loader)

```
convert_dir_to_note_sequences --input_dir="../midi_data_out/piano_filtered" --output_file="../midi_data_out/piano_filtered/notesequences.tfrecord" 
```
