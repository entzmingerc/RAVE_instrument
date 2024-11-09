'''
Train an Autoencoder using the Pulsar dataset
save the trained network parameters

batch size, epochs, learning rate, sample rate, time len, num samples

'''

import torch as t
import numpy as np
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from pulsaroneshots import PulsarOneShotsDataset
from autoencoder import AutoencoderNetwork
from torchvision import models
# from torchinfo import summary
import wandb
import matplotlib
import matplotlib.pyplot as plt
import librosa
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle


WANDB_ENABLE = False
BATCH_SIZE = 10
EPOCHS = 15
PLOT_PER_EPOCH = 5
LEARNING_RATE = 0.0005
ANNOTATIONS_FILE = 'datasets\\PulsarOneShots\\metadata\\PulsarOneShotsKick.csv'
AUDIO_DIR = 'datasets\\PulsarOneShots\\audio'
# TIME_LEN = 2 # seconds desired ish
# NUM_SAMPLES = SAMPLE_RATE * TIME_LEN # samples, width of spectrogram
# input shape to model (batch, channel, num_mels, time_frames)
# time_frames = 1 + (waveform_length - window_length) // hop_length
# 2 seconds of audio gets me a value of 87, but during conv2d it floors the division
# resulting in 87 -> floor(43.5) -> floor(21.5), then decoder outputs 21 -> 42 -> 84, thus can't compare 87 and 84
# instead let's solve time_frames give us 1 + floor(83.XX)
# thus waveform_length = [43520 .. 44032) ... okay that didn't work it gave me 86 output, must be calculating something wrong
# ... guess and check until the tensors work, 44100 - 1600 samples seems to be in range, bit shorter than 2 seconds
# (44100 - 1600) / 22050 = 1.927 sec lol
N_FFT = 512 - 1
N_STFT = int((N_FFT//2) + 1)
SAMPLE_RATE = 44100 # samples / seconds
# NUM_SAMPLES = 42500 # = 44100 - 1600 TO DO, figure out how to calcualte this exactly
NUM_SAMPLES = 41800 # switching to spectrogram 512 fft gives 256 freqs, 164 time frames if I shorten samples to get something divisible by 4
NUM_MELS = 128 # height of spectrogram
NUM_CLASSES = 4 # 0 bass, 1 hat, 2 kick, 3 snare
# model hyperparams
# hyperparameter tuning uhhhh Ray Tune whatever that is
# first we need to make a test dataset lol, not sure how 80% train 20% test

def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    return train_dataloader

def plot_spectrogram(in_spec, title='Spectrogram (db)', ylabel='freq-bin', aspect='auto', xmax=None):
    fig, axis = plt.subplots(1, 1)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.set_xlabel('frame')
    im = axis.imshow(
        librosa.power_to_db(np.abs(in_spec)**2),
        origin='lower',
        aspect=aspect
    )
    if xmax:
        axis.set_xlim((0, xmax))
    fig.colorbar(im, ax=axis)
    # matplotlib.pyplot.show(block=False) # what does block do?
    plt.show(block=False) # what does block do?

def train(model, data_loader, loss_fn, optimizer, device, epochs, log_every):
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}')
        for input_spec, label in data_loader:
            # DO TRAIN STEP
            # zero gradients for tesnors first
            optimizer.zero_grad()

            # forward pass
            input_spec = input_spec.to(device)
            # print(input_spec.shape)
            output_spec = model(input_spec)

            # calculate loss for given sample
            # print(input_spec.shape)
            # print(output_spec.shape)
            loss = loss_fn(output_spec, input_spec)

            # backpropagate error, generate gradients
            loss.backward()

            # update weights
            optimizer.step()

        print(f'loss: {loss.item()}')
        print('---------------------------')
        if WANDB_ENABLE:
            wandb.log({'loss': loss})
        if epoch % log_every == log_every - 1:
            with t.no_grad():
                in_img = data_loader.__iter__().__next__()[0][0, :, :, :].to(device)
                in_img = in_img.unsqueeze(0).to(device)
                out_img = model(in_img)
                # convert to numpy arrays
                in_img = in_img.squeeze().detach().cpu().numpy()
                out_img = out_img.squeeze().detach().cpu().numpy()
                # plt.plot(in_img, out_img)
                # plt.show()
                plot_spectrogram(in_img, title="input (db)")
                plot_spectrogram(out_img, title="output (db)")
    print('Finished training')

if __name__ == '__main__':

    # get device
    if t.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'Using {device}')

    # instantiating mel transform obj, pulsar dataset, training dataloader
    # waveform input and returns Mel spectrogram size (…, n_mels, time)
    # so batch is 1, n_mels = 64, time = 2 seconds?
    # or maybe it'll output samples as time axis?
    # mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    #     sample_rate=SAMPLE_RATE,
    #     n_fft=N_FFT,
    #     n_mels=NUM_MELS)
    normal_spectrogram = torchaudio.transforms.Spectrogram(
        n_fft=N_FFT,
        power=None) # returns complex spectrum if None
    # inverse_spectrogram = torchaudio.transforms.InverseSpectrogram(
    #     n_fft=N_FFT)
    puslar_data = PulsarOneShotsDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            normal_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)
    print(f'num of training samples: {puslar_data.__len__()}')
    train_dataloader = create_data_loader(puslar_data, BATCH_SIZE)

    # construct model and assign it to device
    # latent and hidden dim size are very unknown
    vae = AutoencoderNetwork(latent_dim_size=3, 
                             hidden_dim_size=512).to(device)
    # summary(model=vae, input_size=(1, 256, 164), dtypes=[t.cfloat])  # FAILED TO RUN TORCHINFO WOOOO

    # initialise loss funtion + optimizer
    # loss_fn = nn.MSELoss()
    loss_fn = nn.L1Loss()
    optimizer = t.optim.Adam(vae.parameters(),
                                lr=LEARNING_RATE)
    
    # initialize wandb
    if WANDB_ENABLE:
        wandb.init(
            # Set the project where this run will be logged
            project='basic-intro', 
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=f'tutorial_ae',
            config={
                'epochs' : EPOCHS, 
                'batch_size': BATCH_SIZE, 
                'lr' : LEARNING_RATE, 
                'architecture': 'Autoencoder',
                'dataset': 'PulsarOneShots'
            }
        )

    # train model
    train(vae, train_dataloader, loss_fn, optimizer, device, EPOCHS, PLOT_PER_EPOCH)

    # save model
    t.save(vae.state_dict(), 'vae1.pth')
    print('Trained feed forward net saved at vae1.pth')
    if WANDB_ENABLE:
        wandb.finish()
    
    input("pause: plot show spectrogram input")
    print("END")