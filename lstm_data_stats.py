import sys
sys.path.append('../note_seq') # needed unless installing forked lib from github

import numpy as np
import matplotlib.pyplot as plt
from models.model_lstm import MusicLSTMNet, LSTM
from loaders.dataloader_midi import create_dataset_from_midi, load_midi_to_seq
from training.train_lstm import train_lstm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils.paths as paths
from utils.tools import flatten, AutoVivification
from collections import Counter
from models.distributions import get_histograms_from_dataloader
from tqdm import tqdm

dataset_path = paths.midi_dir

# Set seed such that we always get the same dataset
np.random.seed(42)

# Encoding-Tokenizing parameters
instruments = [0,1]
lead_instrument   = ('melody',instruments[0])
accomp_instrument = ('bass',instruments[1])

extract_names = (lead_instrument[0], accomp_instrument[0])
extract_instruments = [lead_instrument[1], accomp_instrument[1]]

name_instrument_map = {
    extract_names[0]:  extract_instruments[0], 
    extract_names[1]:  extract_instruments[1]
    }


sequences = load_midi_to_seq(dataset_path, name_instrument_map, recursive=False)


#%%
max_chunks_lists = [4,5,6]
steps = [1,2,4]
D = AutoVivification()
for i, spq in enumerate(steps):
    tokenizer_kwargs = {'split_in_bar_chunks': 16, 
                        'steps_per_quarter': spq, 
                        'min_note': 60, 
                        'max_note':72}

    length_per_seq =  tokenizer_kwargs['steps_per_quarter'] * 4 * tokenizer_kwargs['split_in_bar_chunks']
    print('length_per_seq: ', length_per_seq)

    # Dataset creation, splitting and batching parameters
    dataset_split_kwargs = {'p_train': 0.6, 'p_val': 0.3, 'p_test': 0.0,
                            'batch_size': 1,
                            'eval_batch_size': 1}

    train, val, test, t = create_dataset_from_midi(sequences, 
                                                lead_instrument, 
                                                accomp_instrument, 
                                                print_info=False,
                                                **tokenizer_kwargs, 
                                                **dataset_split_kwargs)
    encoder_decoder = t.encoder_decoder
    num_sequences = t.song_count

    #LSTM hyperparameters
    vocab_size = t.vocab_size
    num_epochs = 30 #100
    learning_rate = 1e-4

   

    mel_notes, bass_notes = get_histograms_from_dataloader(test, vocab_size=vocab_size, \
                                                        plot=False)
    #print('Melody histogram: ', Counter(mel_notes))
    #print('Bass histogram: ', Counter(bass_notes))
    D[i]['mel_notes'] = Counter(mel_notes)
    D[i]['bass_notes'] = Counter(bass_notes)
    D[i]['tokenizer'] = t

step_1_mel = []
step_1_bass = []
step_2_mel = []
step_2_bass = []
step_4_mel = []
step_4_bass = []
for i in range(0,vocab_size):
    step_1_mel.append(D[0]['mel_notes'][i])
    step_1_bass.append(D[0]['bass_notes'][i])
    step_2_mel.append(D[1]['mel_notes'][i])
    step_2_bass.append(D[1]['bass_notes'][i])
    step_4_mel.append(D[2]['mel_notes'][i])
    step_4_bass.append(D[2]['bass_notes'][i])
#%%
### Melody Histogram

fig, ax = plt.subplots()

index = np.arange(vocab_size)
bar_width = 0.3

opacity = 0.5

rects1 = plt.bar(index, step_1_mel, bar_width,
                 alpha=opacity,
                 color='b',
                 label='steps_per_quarter=1')

rects2 = plt.bar(index + bar_width, step_2_mel, bar_width,
                 alpha=opacity,
                 color='slategray',
                 label='steps_per_quarter=2')

rects3 = plt.bar(index + 2*bar_width, step_4_mel, bar_width,
                 alpha=opacity,
                 color='r',
                 label='steps_per_quarter=4')

plt.xlabel('Events')
notes = list(range(60,73))
notes = [str(i) for i in notes]
notes.insert(0,'-1')
notes.insert(0,'-2')
notes = tuple(notes)
plt.ylabel('Counts')
plt.title('Melody Histogram')
plt.xticks(index + bar_width, notes)
plt.legend()

plt.tight_layout()
plt.show()
plt.savefig('mel_quantization_stats.png')

#%%
#Bass Histogram
fig, ax = plt.subplots()

index = np.arange(vocab_size)
bar_width = 0.3

opacity = 0.5
error_config = {'ecolor': '0.3'}

rects1 = plt.bar(index, step_1_bass, bar_width,
                 alpha=opacity,
                 color='b',
                 label='steps_per_quarter=1')

rects2 = plt.bar(index + bar_width, step_4_bass, bar_width,
                 alpha=opacity,
                 color='r',
                 label='steps_per_quarter=4')

plt.xlabel('Events')
notes = list(range(60,73))
notes = [str(i) for i in notes]
notes.insert(0,'-1')
notes.insert(0,'-2')
notes = tuple(notes)
plt.ylabel('Counts')
plt.title('Bass Histogram')
plt.xticks(index + bar_width, notes)
plt.legend()

plt.tight_layout()
plt.show()
plt.savefig('bass_quantization_stats.png')

#%%
fig, ax = plt.subplots()

index = np.arange(vocab_size)
bar_width = 0.33

opacity = 0.7
error_config = {'ecolor': '0.3'}

rects1 = plt.bar(index, step_1_mel, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Melody')

rects2 = plt.bar(index + bar_width, step_1_bass, bar_width,
                 alpha=opacity,
                 color='m',
                 label='Bass')

plt.xlabel('Events')
notes = list(range(60,73))
notes = [str(i) for i in notes]
notes.insert(0,'-1')
notes.insert(0,'-2')
notes = tuple(notes)
plt.ylabel('Counts')
plt.title('Melody vs Bass Histogram, step_per_quarter = 1')
plt.xticks(index + bar_width, notes)
plt.legend()

plt.tight_layout()
plt.show()
plt.savefig('melvsbass_spq1.png')

#%%
import numpy as np
import matplotlib.pyplot as plt


vocab_size = 14
step_1_mel = []
step_1_bass = []
step_4_mel = []
step_4_bass = []
for i in range(0,vocab_size):
    step_1_mel.append(D[0]['mel_notes'][i])
    step_1_bass.append(D[0]['bass_notes'][i])
    step_4_mel.append(D[1]['mel_notes'][i])
    step_4_bass.append(D[1]['bass_notes'][i])



fig, axs = plt.subplots(nrows=2,ncols=1)

index = np.arange(vocab_size)
bar_width = 0.33

opacity = 0.5
error_config = {'ecolor': '0.3'}

axs[0].bar(index, step_1_mel, bar_width,
                 alpha=opacity,
                 color='b',
                 label='steps_per_quarter = 1')

axs[0].bar(index + bar_width, step_4_mel, bar_width,
                 alpha=opacity,
                 color='r',
                 label='steps_per_quarter = 4')

axs[0].set_xlabel('Events')
notes = list(range(60,73))
notes = [str(i) for i in notes]
notes.insert(0,'-1')
notes.insert(0,'-2')
notes = tuple(notes)
axs[0].set_ylabel('Counts')
axs[0].set_title('Melody Histogram')
axs[0].set_xticks(index + bar_width, notes)
axs[0].legend()

axs[1].bar(index, step_1_bass, bar_width,
                 alpha=opacity,
                 color='b')

axs[1].bar(index + bar_width, step_4_bass, bar_width,
                 alpha=opacity,
                 color='r')

axs[1].set_xlabel('Events')
notes = list(range(60,73))
notes = [str(i) for i in notes]
notes.insert(0,'-1')
notes.insert(0,'-2')
notes = tuple(notes)
axs[1].set_ylabel('Counts')
axs[1].set_title('Bass Histogram')
axs[1].set_xticks(index + bar_width, notes)
#axs[1].legend()

plt.tight_layout()
plt.show()
plt.savefig('quantization_stats.png')


#%%
max_chunk_lists = list(range(1,30))
steps = [1,4]
D = AutoVivification()
for i, mc in enumerate(max_chunk_lists):
    tokenizer_kwargs = {'split_in_bar_chunks': mc, 
                        'steps_per_quarter': 1, 
                        'min_note': 60, 
                        'max_note':72}

    length_per_seq =  tokenizer_kwargs['steps_per_quarter'] * 4 * tokenizer_kwargs['split_in_bar_chunks']
    print('length_per_seq: ', length_per_seq)

    # Dataset creation, splitting and batching parameters
    dataset_split_kwargs = {'p_train': 0.6, 'p_val': 0.3, 'p_test': 0.0,
                            'batch_size': 1,
                            'eval_batch_size': 1}

    train, val, test, t = create_dataset_from_midi(sequences, 
                                                lead_instrument, 
                                                accomp_instrument, 
                                                print_info=False,
                                                **tokenizer_kwargs, 
                                                **dataset_split_kwargs)
    encoder_decoder = t.encoder_decoder
    num_sequences = t.song_count

    D[i]['train'] = train
    D[i]['val'] = val
    D[i]['test'] = test

index = list(range(1,30))
n_seq = [len(D[i]['test']) for i in D.keys()]
plt.plot(max_chunk_lists, n_seq[:-1], marker='o')
plt.xlabel('split_in_bar_chunks')
#plt.xticks(index, index)
plt.ylabel('# of sequences')
plt.savefig('number_of_sequences.png')