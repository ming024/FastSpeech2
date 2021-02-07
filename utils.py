import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.io import wavfile

import hparams as hp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_mask_from_lengths(lengths, max_len=None):
    if max_len is None:
        max_len = torch.max(lengths).item()
    batch_size = lengths.shape[0]

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def vocoder_infer(mels, vocoder, paths, lengths=None):
    with torch.no_grad():
        wavs = vocoder.inverse(mels / np.log(10)).cpu().numpy() * hp.max_wav_value
    wavs = wavs.astype("int16")
    for i in range(len(mels)):
        wav = wavs[i]
        path = paths[i]
        if lengths is not None:
            length = lengths[i]
            wavfile.write(path, hp.sampling_rate, wav[:length])
        else:
            wavfile.write(path, hp.sampling_rate, wav)


def get_vocoder():
    vocoder = torch.hub.load(
        "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
    )
    vocoder.mel2wav.eval()
    vocoder.mel2wav.to(device)

    return vocoder


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded


def get_speaker_to_id():
    with open(os.path.join(hp.preprocessed_path, "speakers.json")) as f:
        return json.load(f)


def get_pretrained_embedding(speaker, folder):
    return [
        np.load(os.path.join(hp.preprocessed_path, folder, filename))
        for filename in os.listdir(os.path.join(hp.preprocessed_path, folder))
        if speaker in filename
    ]


def get_gst(mels, model):
    ret = []
    for mel in mels:
        mel = torch.from_numpy(np.array([mel])).float().to(device)
        style_embedding = model.module.reference_encoder(mel)
        gst = model.module.style_attention(style_embedding)
        ret.append(gst.detach().cpu().numpy()[0])
    return np.array(ret)