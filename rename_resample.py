import os
import subprocess
from pathlib import Path

# Set directory containing all your .wav files
input_dir = 'PulsarAllSamples'
output_dir = 'PulsarAllSamplesResampled'

# Set the desired sampling rate (44,100 Hz)
sampling_rate = 44100

# Loop through all subfolders and .wav files in the input directory
sample_count = 0
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.endswith('.wav'):
            # Get the original file path and name
            filepath = os.path.join(root, file)

            # Extract the file stem (without extension)
            stem = os.path.splitext(file)[0]
            # resampled_filepath = Path(output_dir) / f'{stem}_resampled.wav'
            resampled_filepath = Path(output_dir) / f'{sample_count}.wav'

            # Resample the .wav file to 44,100 Hz using FFmpeg
            # https://gist.github.com/tayvano/6e2d456a9897f55025e25035478a3a50  FFmpeg incantations, -y enables overwrite
            subprocess.run([
                'ffmpeg', 
                '-y',
                '-i', filepath,
                '-ar', str(sampling_rate),
                '-c:a', 'pcm_s16le',
                '-ac', str(1),
                '-hide_banner',
                '-loglevel', 'error',
                str(resampled_filepath)
            ])
            sample_count = sample_count + 1