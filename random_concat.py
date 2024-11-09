import os
import random
import subprocess

# Set directory containing all your .wav files
input_dir = 'PulsarAllSamplesResampled'
output_dir = 'PulsarAllConcat'

# Define the number of output files, 2653 items, so let's output 7 .wav files each composed of 379 files concatenated
num_output_files = 7

# Get a list of all WAV files in the input directory
wav_files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]

# Loop through each output file
all_files = [*range(2653)]
for i in range(num_output_files):

    # 2653 items, so let's output 7 files each composed of 379 files concatenated
    selected_files = []
    for j in range(379):
        sel = random.choice(all_files)
        selected_files.append(wav_files[sel])
        all_files.remove(sel)

    # Create a temporary text file with the selected file names
    with open(f"{output_dir}\concat.txt", "w") as f:
        for file in selected_files:
            f.write(f"file '{input_dir}\{file}'\n")

    # Construct the command for concatenating the selected files
    cmd = f"ffmpeg -f concat -safe 0 -i {output_dir}\concat.txt -c copy {output_dir}\output_{i}.wav"
    subprocess.run(cmd)
    # Remove the temporary text file
    os.remove(f"{output_dir}\concat.txt")