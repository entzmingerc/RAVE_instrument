"""
This defines the Autoencoder network layers
init, forward
"""

import os
import sys
import torch as t
from torch import nn, optim
import einops
from einops.layers.torch import Rearrange
from tqdm import tqdm
from dataclasses import dataclass, field
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import pandas as pd
from torchsummary import summary

class AutoencoderNetwork(nn.Module):
    def __init__(self,
                 latent_dim_size: int,
                 hidden_dim_size: int):
        super().__init__()
        self.latent_dim_size = latent_dim_size
        self.hidden_dim_size = hidden_dim_size
        # self.num_mels = num_mels
        self.magic_inner_dim_size = 83968
        # self.num_classes = num_classes
        # self.num_channels = num_classes + 1
        # input: num_channels, num_mels, time_frames
        # num_channels usually 1 for single channel audio
        # NUM_MELS = 64
        # NUM_SAMPLES = sample rate * time len, 22050 * 2 = 44100
        # time_frames = 1 + (waveform_length - win_length) // hop_length
        # 1 + (44100 - 1024) // 512 = 85????
        # conv blocks / flatten / linear / softmax
        # kernel_size 4, stride 2, padding 1 will cut dimensions in half I think
        # need to know size of input spectrogram tensor to calculate nn.Linear layers
        # length of time most significant driver of parameter count, this is 2 sec
        # hidden_dim_size is hyperparameter, grid search, trial and error
        # batch, channels+class, num_mels, time_frames
        # input        = 1,  1, 64, 84
        # after conv 1 = 1, 16, 32, 42
        # after conv 2 = 1, 32, 16, 21
        # flatten      = 32 * 16 * 21 = 10,752
        # linear       = in 10,752 out hidden_dim_size
        # n_mels = 64, and time frames = 84 and channels = 1, 16, 32
        # n_mels/4 * timeframes/4 * 32 = magic number
        # SPECTROGRAMS (not mels) trying 512 nfft and 41800 samples, decreasing nfft doesn't affect much, just most time resolution tradeoff
        # which honestly is perfectly fine for this
        # COMPLEX NUMBERS by specifiying power=None in spectrum() to preserve phase information for reconstruction needed
        # input        = 1,  1, 256, 164
        # after conv 1 = 1, 16, 128, 82
        # after conv 2 = 1, 32, 64, 41
        # flatten      = 32 * 64 * 41 = 83968
        # linear       = in 83968 out hidden_dim_size
        # shape of input is like (batches, channels=1 for mono, freq bins(?), time frames)
        # and each value is complex number, idk if that extends dimensions yet
        # [(W−K+2P)/S]+1
        # 2d convolution output shape
        # output_shape = (batch_size, out_channels, output_height, output_width)
        # output_height is calculated as: output_height = (num_mels + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
        # output_width is calculated as: output_width = (time_frames + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
        # self.encoder = nn.Sequential(
        self.conv1 = nn.Conv2d(                
                in_channels=1,
                out_channels=16,
                kernel_size=4,
                stride=2,
                padding=1,
                dtype=t.cfloat
            )
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
                dtype=t.cfloat
            )
        self.relu2 = nn.ReLU()
        self.flatten1 = nn.Flatten()
        self.linear1 = nn.Linear(self.magic_inner_dim_size, hidden_dim_size, dtype=t.cfloat)
        self.relu3 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim_size, latent_dim_size, dtype=t.cfloat)
        # self.decoder = nn.Sequential(
        self.linear3 = nn.Linear(latent_dim_size, hidden_dim_size, dtype=t.cfloat)
        self.relu4 = nn.ReLU()
        self.linear4 = nn.Linear(hidden_dim_size, self.magic_inner_dim_size, dtype=t.cfloat)
        self.relu5 = nn.ReLU()
        self.einreshape = einops.layers.torch.Rearrange("b (c h w) -> b c h w", c=32, h=64, w=41)
        self.conv3 = nn.ConvTranspose2d(
                in_channels=32, 
                out_channels=16, 
                kernel_size=4, 
                stride=2, 
                padding=1,
                dtype=t.cfloat
            )
        self.relu6 = nn.ReLU()
        self.conv4 = nn.ConvTranspose2d(
                in_channels=16, 
                out_channels=1, 
                kernel_size=4, 
                stride=2, 
                padding=1,
                dtype=t.cfloat
            )

    def forward(self, input_data: t.Tensor) -> t.Tensor:        
        # clamp (relu) not supported for complex types, need to clamp real and img separately then recombine
        # x = self.encoder(input_data)
        x = self.conv1(input_data)
        # x = self.relu1(x)
        x = self.relu1(x.real).type(t.complex64) + 1j * self.relu1(x.imag).type(t.complex64)
        x = self.conv2(x)
        # x = self.relu2(x)
        x = self.relu1(x.real).type(t.complex64) + 1j * self.relu1(x.imag).type(t.complex64)
        x = self.flatten1(x)
        x = self.linear1(x)
        # x = self.relu3(x)
        x = self.relu1(x.real).type(t.complex64) + 1j * self.relu1(x.imag).type(t.complex64)
        x = self.linear2(x)
        
        # x = self.decoder(x)
        x = self.linear3(x)
        # x = self.relu4(x)
        x = self.relu1(x.real).type(t.complex64) + 1j * self.relu1(x.imag).type(t.complex64)
        x = self.linear4(x)
        # x = self.relu5(x)
        x = self.relu1(x.real).type(t.complex64) + 1j * self.relu1(x.imag).type(t.complex64)
        x = self.einreshape(x)
        x = self.conv3(x)
        # x = self.relu6(x)
        x = self.relu1(x.real).type(t.complex64) + 1j * self.relu1(x.imag).type(t.complex64)
        x = self.conv4(x)
        return x
        # input: num_channels, num_mels, time_frames
        # concatenate the onehotlabel to the input tensor
        # possibly allows learning multiple classes of sounds
        # one_hot_label = t.eye(self.num_classes)[class_label].to(input_data.device)
        # one_hot_label = one_hot_label.unsqueeze(-1).unsqueeze(-1)   
        # one_hot_label = one_hot_label.repeat(1, 1, input_data.shape[2], input_data.shape[3])
        # input_tensor = t.cat((input_data, one_hot_label), dim=1)

if __name__ == "__main__":
    pass