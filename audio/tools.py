import torch
import numpy as np
from scipy.io.wavfile import write

from audio.audio_processing import griffin_lim


def get_mel_from_wav(audio, _stft):
    audio = torch.clip(torch.FloatTensor(audio).unsqueeze(0), -1, 1)
    audio = torch.autograd.Variable(audio, requires_grad=False)
    melspec, energy = _stft.mel_spectrogram(audio)
    melspec = torch.squeeze(melspec, 0).numpy().astype(np.float32)
    energy = torch.squeeze(energy, 0).numpy().astype(np.float32)

    return melspec, energy


def inv_mel_spec(mel, out_filename, _stft, griffin_iters=60):
    mel = torch.stack([mel])
    mel_decompress = _stft.spectral_de_normalize(mel)
    mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
    spec_from_mel_scaling = 1000
    spec_from_mel = torch.mm(mel_decompress[0], _stft.mel_basis)
    spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
    spec_from_mel = spec_from_mel * spec_from_mel_scaling

    audio = griffin_lim(
        torch.autograd.Variable(spec_from_mel[:, :, :-1]), _stft._stft_fn, griffin_iters
    )

    audio = audio.squeeze()
    audio = audio.cpu().numpy()
    audio_path = out_filename
    write(audio_path, _stft.sampling_rate, audio)
