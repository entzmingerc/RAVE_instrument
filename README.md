# RAVE_instrument
(WIP) This is my work studying RAVE models and learning SuperCollider. Currently it's set up to read audio input from an audio interface and process it through different [RAVE](<https://forum.ircam.fr/projects/detail/rave/>) models. Whenever the `createNode` function is called, if the labeled node doesn't exisit, it creates a node with the loaded RAVE model on a separate audio bus and the audio input is sent to that bus. If the node is created, it frees the node. This lets me turn on and off models easily for live coding. On my machine, I can run about 10 models and the input audio stream on separate audio buses before getting significant voice culling. 

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