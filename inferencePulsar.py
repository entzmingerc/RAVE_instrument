"""
Load the parameters of a trained network
use it for inference
evaluate the results
"""
import os
import io
import random
import torch as t
import torchaudio
import matplotlib.pyplot as plt
from autoencoder import AutoencoderNetwork
from pulsaroneshots import PulsarOneShotsDataset
from trainPulsar import AUDIO_DIR, ANNOTATIONS_FILE, SAMPLE_RATE, NUM_SAMPLES, N_FFT, NUM_MELS, plot_spectrogram

PLOT_ENABLE = False

class_mapping = [
    "bass",
    "hat",
    "kick",
    "snare"
]

def evaluate(model, in_spec):
    with t.no_grad():
        model.eval() # set it to evaluation
        out_spec = model(in_spec) # input spec, output spec
    return out_spec

def audioAttenuate(v, new_minmax):
    # normalizes between +/- new_minmax for audio file
    # this preserves asynchronous waveforms
    # I just want to attenuate whatever waveform down to +/- 1
    v_min = v.min()
    v_max = v.max()
    if abs(v_max) > abs(v_min):
        # what * abs(v_max) = new_minmax
        scale = new_minmax / abs(v_max)
    else:
        scale = new_minmax / abs(v_min)
    return v * scale

if __name__ == "__main__":
    # get device
    if t.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    # load the trained model    
    vae = AutoencoderNetwork(latent_dim_size=3, 
                             hidden_dim_size=512,
                             num_mels=NUM_MELS).to(device)
    state_dict = t.load("vae1.pth")
    vae.load_state_dict(state_dict)

    # make the mel spectrogram data transform
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        n_mels=NUM_MELS
    )

    # use transform to create dataloader for testing
    pulsardata = PulsarOneShotsDataset(
        ANNOTATIONS_FILE,
        AUDIO_DIR,
        mel_spectrogram,
        SAMPLE_RATE,
        NUM_SAMPLES,
        device
    )

    # create transforms mel2spec, spec2audio  
    invMel = torchaudio.transforms.InverseMelScale(n_stft=N_FFT, n_mels=NUM_MELS).to(device)
    invSpec = torchaudio.transforms.InverseSpectrogram(n_fft=N_FFT, onesided=False).to(device)

    in_mel = []
    out_mel = []

    # generate and save 8 wav files
    for i in range(8):
        # get a sample from dataset 
        randSample = random.randint(0, pulsardata.__len__())
        in_mel = pulsardata[randSample][0].unsqueeze_(0) # 1, 64, 84 (channels, n_mels, time_frames), unsqueeze adds an extra dimension to the tensor 1, 1, 64, 84
        out_mel = evaluate(vae, in_spec=in_mel) # send in a sample through encoder+decoder, output a spectrogram

        # transform mel spectrogram to log power spectrogram 
        in_spec = invMel(in_mel)
        out_spec = invMel(out_mel)

        # transform spectrogram to audio
        in_wav = invSpec(in_spec.type(t.complex64), NUM_SAMPLES)
        out_wav = invSpec(out_spec.type(t.complex64), NUM_SAMPLES)

        # plot the input, plot the output
        if PLOT_ENABLE:
            in_spec = in_spec.squeeze().cpu().numpy()    
            out_spec = out_spec.squeeze().cpu().numpy()
            plot_spectrogram(in_spec, title='input spectrogram (db)')
            plot_spectrogram(out_spec, title='output spectrogram (db)')

        # attenuate to -1 and +1 amplitude for .wav file
        # .wav files are between -32768 to +32767, but torch likes things -1 and 1, torch.load does -1 to 1
        in_wav_normalized = audioAttenuate(in_wav[0].detach().cpu(), 1)
        out_wav_normalized = audioAttenuate(out_wav[0].detach().cpu(), 1)

        # save to file
        torchaudio.save(f'generatedaudio\\in-{i}.wav',  src=in_wav_normalized,  sample_rate=SAMPLE_RATE, format="wav") # possible to mp3
        torchaudio.save(f'generatedaudio\\out-{i}.wav', src=out_wav_normalized, sample_rate=SAMPLE_RATE, format="wav") # possible to mp3
        print(f"saved in & out samples {randSample}")


    if PLOT_ENABLE:
        input("...hey check out those plots yo")
    print("END")

    # # TEST 
    # test_sample_rate = 16000
    # test_num_samples = 16000
    # test_audio_tensor = t.sin(2 * t.pi * t.linspace(0, 1, test_num_samples)).unsqueeze(0)
    # torchaudio.save("generatedaudio\\test.wav", test_audio_tensor, test_sample_rate)
