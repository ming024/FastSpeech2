import torch
import numpy as np
from scipy.io.wavfile import write
from scipy.signal import lfilter

import hparams as hp
import audio.stft as stft
from audio.audio_processing import griffin_lim

_stft = stft.TacotronSTFT(
    hp.filter_length,
    hp.hop_length,
    hp.win_length,
    hp.n_mel_channels,
    hp.sampling_rate,
    hp.mel_fmin,
    hp.mel_fmax,
)


def get_mel_from_wav(audio):
    sampling_rate = hp.sampling_rate
    if sampling_rate != _stft.sampling_rate:
        raise ValueError(
            "{} {} SR doesn't match target {} SR".format(
                sampling_rate, _stft.sampling_rate
            )
        )

    audio = lfilter([1, -hp.preemph], [1], audio)
    audio = torch.clip(torch.FloatTensor(audio).unsqueeze(0), -1, 1)
    audio = torch.autograd.Variable(audio, requires_grad=False)
    melspec, energy = _stft.mel_spectrogram(audio)
    melspec = torch.squeeze(melspec, 0)
    energy = torch.squeeze(energy, 0)

    return melspec, energy


def inv_mel_spec(mel, out_filename, griffin_iters=60):
    mel = torch.stack([mel])
    # mel = torch.stack([torch.from_numpy(_denormalize(mel.numpy()))])
    mel_decompress = _stft.spectral_de_normalize(mel)
    mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
    spec_from_mel_scaling = 1000
    spec_from_mel = torch.mm(mel_decompress[0], _stft.mel_basis)
    spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
    spec_from_mel = spec_from_mel * spec_from_mel_scaling

    audio = griffin_lim(
        torch.autograd.Variable(
            spec_from_mel[:, :, :-1]), _stft.stft_fn, griffin_iters
    )

    audio = audio.squeeze()
    audio = audio.cpu().numpy()
    audio_path = out_filename
    write(audio_path, hp.sampling_rate, audio)
