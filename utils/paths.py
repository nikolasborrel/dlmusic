import os
main_dir = os.getcwd()  #the default path should be dlmusic
root_dir = main_dir[0:-len('/dlmusic')]
model_serialized_dir = root_dir + '/dlmusic_data/serialized_models/'
midi_dir = root_dir + '/dlmusic_data/CLEAN_MIDI_BM_small/'
