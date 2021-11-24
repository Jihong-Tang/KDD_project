import os
import json
import random
import pickle
import numpy as np

from torch.utils.data.dataset import Dataset

audio_data = [
    'lcmfMfcc.breathing-deep.dat',
    'lcmfMfcc.breathing-shallow.dat',
    'lcmfMfcc.cough-heavy.dat',
    'lcmfMfcc.cough-shallow.dat',
    'lcmfMfcc.counting-fast.dat',
    'lcmfMfcc.counting-normal.dat',
    'lcmfMfcc.vowel-a.dat',
    'lcmfMfcc.vowel-e.dat',
    'lcmfMfcc.vowel-o.dat'
]
meta_info = 'metadata.json'


class CoswaraDataset(Dataset):

    def __init__(self, base_dir: str, seq_len: int = 96):
        super(CoswaraDataset, self).__init__()

        self.audios = []
        self.labels = []
        self.seq_len = seq_len
        self.freq_bin = 26

        print('Loading coswara dataset: {}'.format(base_dir))

        audios_file = 'audios.pkl'
        labels_file = 'labels.pkl'

        if not os.path.exists(audios_file) or not os.path.exists(labels_file):
            for record_date in os.listdir(base_dir):
                date_dir = os.path.join(base_dir, record_date)
                # ignore file, e.g., csv
                if os.path.isdir(date_dir):
                    date_dir = os.path.join(base_dir, record_date, record_date)
                    for participant in os.listdir(date_dir):
                        cur_dir = os.path.join(date_dir, participant)
                        # ignore temp file, e.g., .DS_Store
                        if os.path.isdir(cur_dir):
                            print('Participant: {}'.format(participant))

                            # get label
                            covid_status_label = 0
                            with open(os.path.join(cur_dir, meta_info)) as f:
                                data = json.load(f)
                                covid_status = data['covid_status']
                                if covid_status == 'healthy':
                                    covid_status_label = 1

                            # get audio data
                            current_audios = []
                            current_labels = []
                            fg = True
                            for audio_file in audio_data:
                                data = np.genfromtxt(
                                    os.path.join(cur_dir, audio_file),
                                    delimiter=','
                                )
                                print(data.shape)
                                if data.shape[0] >= self.seq_len:
                                    current_audios.append(data)
                                    current_labels.append(covid_status_label)
                                else:
                                    fg = False
                                    print('The audio frame length is shorter than {}, not included'.format(
                                        self.seq_len))
                                    break
                            if fg:
                                self.audios.extend(current_audios)
                                self.labels.extend(current_labels)
                            print(covid_status_label)
                            # break
            with open(audios_file, 'wb') as f:
                pickle.dump(self.audios, f)
            with open(labels_file, 'wb') as f:
                pickle.dump(self.labels, f)
        else:
            self.audios = pickle.load(open(audios_file, 'rb'))
            self.labels = pickle.load(open(labels_file, 'rb'))

    def __getitem__(self, index):
        if self.seq_len > 0:
            t = random.randint(0, self.audios[index].shape[0] - self.seq_len)
            return self.audios[index][t: t+self.seq_len, :], self.labels[index]
        else:
            return self.audios[index], self.labels[index]

    def __len__(self):
        return len(self.audios)
