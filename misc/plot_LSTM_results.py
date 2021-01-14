from utils.tools import AutoVivification
import os
import json
import matplotlib.pyplot as plt
results_path = '/home/sarantos/Documents/Music_AI/dlmusic_data/serialized_models/4bars/'

D = AutoVivification()

for root, subdirs, files in os.walk(results_path):
    pass
print(files)

fig, axs = plt.subplots(nrows=2, ncols=1)
for i, f in enumerate(files):
    with open(results_path + f) as json_file:
        D[i] = json.load(json_file)
        axs[0].plot(list(range(1,26)), D[i]['training_loss'], label=f[:-5])
        axs[1].plot(list(range(1,26)), D[i]['validation_loss'])

axs[0].set_title('LSTM performance for seq_len=16, lr=1e-4, batch_size=16')
#axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Training Loss')
axs[0].legend()


#plt.title('LSTM performance for seq_len=16')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Validation Loss')
#plt.legend()
#plt.savefig('Validation Loss')
plt.savefig('4barLSTMperformance.png')


