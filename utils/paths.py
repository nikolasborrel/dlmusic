import os
main_dir = os.getcwd()  #the default path should be dlmusic
root_dir = main_dir[0:-len('/dlmusic')]
model_serialized_dir = root_dir + '/dlmusic_data/serialized_models/'
midi_dir_small = root_dir + '/dlmusic_data/midi_data/CLEAN_MIDI_BM_small_working/'
midi_dir = root_dir + '/dlmusic_data/midi_data/CLEAN_MIDI_BM/'
midi_dir_one_song = root_dir + '/dlmusic_data/midi_data/ONE_SONG/'
