"""
PulsarOneShots Dataset class
read in numbers, get items, preprocessing methods
"""

import os
import torch as t
from torch.utils.data import Dataset
import pandas as pd
import torchaudio


class PulsarOneShotsDataset(Dataset):
    # read in setup parameters
    def __init__(self,
                 annotations_file,
                 audio_dir,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device):
        self.device = device
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
    
    # number of samples
    def __len__(self):
        return len(self.annotations)
      
    # reads a sample, processes it, returns sample and label
    def __getitem__(self, index):
        # get sample and label
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path) # might be weirdly normalizing?

        # preprocessing
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._gain_match_if_necessary(signal) # preserves asynchronous waveforms
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label
    
        # I THINK I DO THE LABEL ENCODING HERE INSTEAD OF IN THE FORWARD FUNCTION
        # sample = {'spec': signal, 'label': label}
        # return sample
    # adjust gain so max amplitude is between -1/+1
    def _gain_match_if_necessary(self, signal):
        v_min = signal.min()
        v_max = signal.max()
        min_amplitude = 0.00000001
        if abs(v_max) > min_amplitude and abs(v_min) > min_amplitude:
            if abs(v_max) > abs(v_min):
                # what times abs(v_max) gets us new_minmax which is 1
                scale = 1 / abs(v_max)
            else:
                scale = 1 / abs(v_min)
        else:
            scale = 0 # only happens if .wav is literally zero
        return signal * scale
    
    # cut samples that are too long
    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    # adds zeros to the end of sample to get to correct size
    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = t.nn.functional.pad(signal, last_dim_padding)
        return signal

    # resamples at correct sample rate if isn't correct
    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            resampler = resampler.to(self.device)
            signal = resampler(signal)
        return signal

    # mean the amplitude, convert stereo to mono
    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = t.mean(signal, dim=0, keepdim=True)
        return signal

    # get path object for a given index
    def _get_audio_sample_path(self, index):
        folder = f"{self.annotations.iloc[index, 2]}"
        path = os.path.join(self.audio_dir, folder, self.annotations.iloc[index, 0])
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 1] # number of class, not string of class