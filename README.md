# VAE Drum Sample Generator
This work began with self-teaching myself PyTorch while building a variational autoencoder. The goal was to train the VAE using spectrogram images of 2 second long drum samples I recorded. I compiled about 2.5 hours worth of drum machine samples that I recorded and using FFmpeg was able to cut the lengths down to size, normalize the left and right audio channels, normalize the amplitude of each waveform, filter out as much white noise as I could, and then store them in a folder. To train the VAE, I converted each .wav sample into a spectrogram first then used the image as an input to the model. The VAE tried to learn the data present in the spectrogram image and recreate it on its output. One problem I ran into immediately was that using all real numbers during training resulted in the phase information of the frequency content of the training sample to be lost when converted into a spectrogram image. Similarly, when attempting to invert the process by transforming a spectrogram back into an audio file, the absent phase data resulted in audible distortion and noise on the generated .wav file. The solution seemed to be either use complex numbers to represent the frequency information in PyTorch, or to find a training method that preserves the phase information somehow. Eventually, I found RAVE which seemed to do everything I was trying to do and more. 

# RAVE_instrument
This is my work in progress for studying RAVE models and learning SuperCollider. Currently it's set up to read audio input from an audio interface and process it through different [RAVE](<https://forum.ircam.fr/projects/detail/rave/>) models. Whenever the `createNode` function is called, if the labeled node doesn't exisit, it creates a node with the loaded RAVE model on a separate audio bus and the audio input is sent to that bus. If the node is created, it frees the node. This lets me turn on and off models easily for live coding. On my machine, I can run about 10 models and the input audio stream on separate audio buses before getting significant voice culling. 

# Requirements
[SuperCollider](<https://supercollider.github.io/>)  

[nn.ar](<https://github.com/elgiano/nn.ar>) from elgiano. This is a SuperCollider extension which allows for loading RAVE models, calling forward, encode, and decode methods on the loaded models, and manipulating latent variables. 

### Pre-Trained Models
- ACIDS-IRCAM: [Download Page](<https://acids-ircam.github.io/rave_models_download>)
- Intelligent Instruments Lab (IIL): 17 models available at [Hugging Face Page](<https://huggingface.co/Intelligent-Instruments-Lab/rave-models>)
- Information about training your own models available at [ACIDS-IRCAM RAVE GitHub page](<https://github.com/acids-ircam/rave>). Checkpoint models for transfer learning are provided by IIL in their Hugging Face repository. 

# Observations
I was interested in timbre transformation and using these models as audio FX, similar to a guitar pedal or plugin. When using RAVE models real-time timbre transfer in a live music context, I found adjusting the amplitude of the input audio and using graphic equalizers (or bandpass filters in general) to control the input harmonic content as relatively easy way to steer the output of the RAVE model without messing with the latent space variabels.  
 
Generally, the models generate different output audio depending on the amplitude profile and frequency spectrum of the audio input. The sound generated from processing a basic square wave synth holding a note for a long period of time sounds different from the same synth inputting shorter plucky notes. It's easy to hear this with some of the vocal models where short plucks start sounding like beat boxing and long held notes are similar to a singing voice. The shorter note isn't just a shortened version of the longer note audio as you'd get with a synthesizer, it's a different spectrum of sound. Similarly, modulating the harmonic content of the input at different rates can generate a variety of changes on the output of the RAVE model. Wavetable synthesizers that sweep through a diverse set of spectrums can result in some really cool sounds on the output.  

TODO: midi sequence the latent space / use priors
