import torch
import torchaudio
import torch.nn as nn
import pandas as pd
import numpy as np



class Data(torch.utils.data.Dataset):

   # this makes it easier to be ovveride in argparse
    parameters = {
        "sample_rate": 8000, "n_feats": 81,
        "specaug_rate": 0.5, "specaug_policy": 3,
        "time_mask": 70, "freq_mask": 15
    }

    def __init__(self, json_path, sample_rate, n_feats, specaug_rate, specaug_policy,
                time_mask, freq_mask, valid=False, shuffle=True, text_to_int=True, log_ex=True):
        self.log_ex = log_ex
        self.text_process = TextProcess()

        print("Loading data json file from", json_path)
        self.data = pd.read_json(json_path, lines=True)

        if valid:
            self.audio_transforms = torch.nn.Sequential(
                LogMelSpec(sample_rate=sample_rate, n_mels=n_feats,  win_length=160, hop_length=80)
            )
        else:
            self.audio_transforms = torch.nn.Sequential(
                LogMelSpec(sample_rate=sample_rate, n_mels=n_feats,  win_length=160, hop_length=80),
                SpecAugment(specaug_rate, specaug_policy, freq_mask, time_mask)
            )


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        try:
            file_path = self.data.key.iloc[idx]
            waveform, _ = torchaudio.load(file_path)
            label = self.text_process.text_to_int_sequence(self.data['text'].iloc[idx])
            spectrogram = self.audio_transforms(waveform) # (channel, feature, time)
            spec_len = spectrogram.shape[-1] // 2
            label_len = len(label)
            if spec_len < label_len:
                raise Exception('spectrogram len is bigger then label len')
            if spectrogram.shape[0] > 1:
                raise Exception('dual channel, skipping audio file %s'%file_path)
            if spectrogram.shape[2] > 1650:
                raise Exception('spectrogram to big. size %s'%spectrogram.shape[2])
            if label_len == 0:
                raise Exception('label len is zero... skipping %s'%file_path)
        except Exception as e:
            if self.log_ex:
                print(str(e), file_path)
            return self.__getitem__(idx - 1 if idx != 0 else idx + 1)
        return spectrogram, label, spec_len, label_len

    def describe(self):
        return self.data.describe()


def collate_fn_padd(data):
    '''
    Padds batch of variable length
    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    # print(data)
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    for (spectrogram, label, input_length, label_length) in data:
        if spectrogram is None:
            continue
       # print(spectrogram.shape)
        spectrograms.append(spectrogram.squeeze(0).transpose(0, 1))
        labels.append(torch.Tensor(label))
        input_lengths.append(input_length)
        label_lengths.append(label_length)

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    input_lengths = input_lengths
    # print(spectrograms.shape)
    label_lengths = label_lengths
    # ## compute mask
    # mask = (batch != 0).cuda(gpu)
    # return batch, lengths, mask
    return spectrograms, labels, input_lengths, label_lengths




