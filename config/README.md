# Config
Here are the config files used to train the single/multi-speaker TTS models.
5 different configurations are given:
- LJSpeech: suggested configuration for LJSpeech dataset.
- LibriTTS: suggested configuration for LibriTTS dataset.
- AISHELL3: suggested configuration for AISHELL-3 dataset.
- LJSpeech_paper: closed to the setting proposed in the original FastSpeech 2 paper.
- YourLanguage: A generic configuration intended to have values replaced for supporting a new language.

Some important hyper-parameters are explained here.

## preprocess.yaml
- **path.lexicon_path**: the lexicon (which maps words to phonemes) used by Montreal Forced Aligner. 
  We provide an English lexicon and a Mandarin lexicon. 
  Erhua (ㄦ化音) is handled in the Mandarin lexicon.
- **mel.stft.mel_fmax**: set it to 8000 if HiFi-GAN vocoder is used, and set it to null if MelGAN is used.
- **pitch.feature & energy.feature**: the original paper proposed to predict and apply frame-level pitch and energy features to the inputs of the TTS decoder to control the pitch and energy of the synthesized utterances. 
  However, in our experiments, we find that using phoneme-level features makes the prosody of the synthesized utterances more natural.
- **pitch.normalization & energy.normalization**: to normalize the pitch and energy values or not. 
  The original paper did not normalize these values.
- **text.use_spe_features**: This will attempt to convert your input text into a sequence of multihot phonological feature vectors which are then cached. This is for normalizing input space dimensions in pre-training/fine-tuning pipelines.
- **text.spe_feature_dim**: Sets the size of phonological feature vectors
- **speaker.embedding & speaker.pretrained_path**: If set to 'deep-speaker', then a path to a [pretrained deep-speaker model](https://drive.google.com/drive/folders/18h2bmsAWrqoUMsh_FQHDDxp7ioGpcNBa) must be used. This will generate a speaker embedding for each input file and then cache an averaged speaker vector for each speaker. It is an alternative to the one-hot speaker embedding system which requires no pre-processing.

## train.yaml
- **optimizer.grad_acc_step**: the number of batches of gradient accumulation before updating the model parameters and call optimizer.zero_grad(), which is useful if you wish to train the model with a large batch size but you do not have sufficient GPU memory.
- **optimizer.anneal_steps & optimizer.anneal_rate**: the learning rate is reduced at the **anneal_steps** by the ratio specified with **anneal_rate**.

## model.yaml
- **transformer.decoder_layer**: the original paper used a 4-layer decoder, but we find it better to use a 6-layer decoder, especially for multi-speaker TTS.
- **variance_embedding.pitch_quantization**: when the pitch values are normalized as specified in ``preprocess.yaml``, it is not valid to use log-scale quantization bins as proposed in the original paper, so we use linear-scaled bins instead. 
- **multi_speaker**: to apply a speaker embedding table to enable multi-speaker TTS or not.
- **vocoder.speaker**: should be set to 'universal' if any dataset other than LJSpeech is used.
- **transformer.spe_features & transformer.spe_feature_dim**: Sets whether using multi-hot phonological feature vectors as inputs and their size.
- **transformer.depthwise_convolutions**: uses depthwise separable convolutions as discussed in Luo et. al. 2021 LightSpeech paper.
- **variance_predictor.use_energy_predictor**: sets whether energy predictor is used or not.
- **use_postnet**: whether to use residual postnet or not.
- **multi_speaker**: whether to use multi-speaker data. If 'one-hot', no preprocessing is required, if 'vector', then preprocess.yaml must be configured to create deep-speaker vectors for each speaker. Currently the speaker embeddings are summed with the output of the encoder, but future work should allow for a speaker variance adaptor architecture.