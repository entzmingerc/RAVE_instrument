import torch as t
import torchaudio
import os

# adjust gain so max amplitude is between -1/+1
def gain_match_if_necessary( signal):
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
def cut_if_necessary( signal):
    if signal.shape[1] > num_samples:
        signal = signal[:, :num_samples]
    return signal

# adds zeros to the end of sample to get to correct size
def right_pad_if_necessary( signal):
    length_signal = signal.shape[1]
    if length_signal < num_samples:
        num_missing_samples = num_samples - length_signal
        last_dim_padding = (0, num_missing_samples)
        signal = t.nn.functional.pad(signal, last_dim_padding)
    return signal

# resamples at correct sample rate if isn't correct
def resample_if_necessary( signal, sr):
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        resampler = resampler
        signal = resampler(signal)
    return signal

# mean the amplitude, convert stereo to mono
def mix_down_if_necessary( signal):
    if signal.shape[0] > 1:
        signal = t.mean(signal, dim=0, keepdim=True)
    return signal

SAMPLE_RATE = 44100
N_FFT = 512 - 1
NUM_MELS = 128
num_samples = 41800
NUM_SAMPLES = num_samples
target_sample_rate = SAMPLE_RATE
audio_sample_path = 'testaudio\\2-148.wav'

# get sample and label
signal, sr = torchaudio.load(audio_sample_path) # might be weirdly normalizing?
# save immediately to test
torchaudio.save(f'testaudio\\out-immediate.wav', src=signal, sample_rate=sr, format="wav")

# preprocessing
signal = resample_if_necessary(signal, sr)
torchaudio.save(f'testaudio\\out-resample.wav', src=signal, sample_rate=SAMPLE_RATE, format="wav")
signal = mix_down_if_necessary(signal)
torchaudio.save(f'testaudio\\out-mixdown.wav', src=signal, sample_rate=SAMPLE_RATE, format="wav")
signal = gain_match_if_necessary(signal) # preserves asynchronous waveforms
torchaudio.save(f'testaudio\\out-gainmatch.wav', src=signal, sample_rate=SAMPLE_RATE, format="wav")
signal = cut_if_necessary(signal)
torchaudio.save(f'testaudio\\out-cut.wav', src=signal, sample_rate=SAMPLE_RATE, format="wav")
signal = right_pad_if_necessary(signal)
torchaudio.save(f'testaudio\\out-rightpad.wav', src=signal, sample_rate=SAMPLE_RATE, format="wav")


# create transforms mel2spec, spec2audio  
N_STFT = int((N_FFT//2) + 1)
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    n_mels=NUM_MELS,
    power=None
)
normal_spectrogram = torchaudio.transforms.Spectrogram(
    n_fft=N_FFT,
    power=None) # returns complex spectrum if None
inverse_spectrogram = torchaudio.transforms.InverseSpectrogram(
    n_fft=N_FFT
    )
mel_scale = torchaudio.transforms.MelScale(n_mels=NUM_MELS, sample_rate=SAMPLE_RATE, n_stft=N_STFT)
inverse_mel = torchaudio.transforms.InverseMelScale(n_stft=N_STFT, n_mels=NUM_MELS)
grifflim_transform = torchaudio.transforms.GriffinLim(n_fft=N_FFT)

print(signal.shape)
# signal = mel_spectrogram(signal)
# print(signal.shape)
# in_spec = inverse_mel(signal)
in_spec = normal_spectrogram(signal)
print(in_spec.shape)
# in_mel = mel_scale(in_spec)
# print(in_mel.shape)
# in_invmel = inverse_mel(in_mel)
# print(in_invmel.shape)
in_wav = inverse_spectrogram(in_spec)
print(in_wav.shape)
torchaudio.save(f'testaudio\\out-inversespectrogram.wav', src=in_wav, sample_rate=SAMPLE_RATE, format="wav")

in_wav_normalized = gain_match_if_necessary(in_wav)
torchaudio.save(f'testaudio\\out-invSpecNORM.wav', src=in_wav_normalized, sample_rate=SAMPLE_RATE, format="wav")

# invMel = torchaudio.transforms.InverseMelScale(
#     n_stft=N_FFT // 2 + 1, 
#     n_mels=NUM_MELS)

# grifflim_transform = torchaudio.transforms.GriffinLim(n_fft=N_FFT)

# transform mel spectrogram to log power spectrogram 

# transform spectrogram to audio
# in_wav = invSpec(in_spec.type(t.complex64), NUM_SAMPLES)