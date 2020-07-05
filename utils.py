import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import os

import text
import hparams as hp

def get_alignment(tier):
    sil_phones = ['sil', 'sp', 'spn', '']

    phones = []
    durations = []
    start_time = 0
    end_time = 0
    end_idx = 0
    for t in tier._objects:
        s, e, p = t.start_time, t.end_time, t.text

        # Trimming leading silences
        if phones == [] and p in sil_phones:
            start_time = e
            continue
        else:
            if p not in sil_phones:
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                phones.append('$')
            durations.append(int(e*hp.sampling_rate/hp.hop_length)-int(s*hp.sampling_rate/hp.hop_length))

    # Trimming tailing silences
    phones = phones[:end_idx]
    durations = durations[:end_idx]
    
    return phones, durations, start_time, end_time

def process_meta(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        text = []
        name = []
        for line in f.readlines():
            n, t = line.strip('\n').split('|')
            name.append(n)
            text.append(t)
        return name, text


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def plot_data(data, titles=None, filename=None):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]

    def add_axis(fig, old_ax, offset=0):
        ax = fig.add_axes(old_ax.get_position(), anchor='W')
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):
        spectrogram, pitch, energy = data[i]
        axes[i][0].imshow(spectrogram, origin='lower')
        axes[i][0].set_aspect(2.5, adjustable='box')
        axes[i][0].set_ylim(0, hp.n_mel_channels)
        axes[i][0].set_title(titles[i], fontsize='medium')
        axes[i][0].tick_params(labelsize='x-small', left=False, labelleft=False) 
        axes[i][0].set_anchor('W')
        
        ax1 = add_axis(fig, axes[i][0])
        ax1.plot(pitch, color='tomato')
        ax1.set_xlim(0, spectrogram.shape[1])
        ax1.set_ylim(0, hp.f0_max)
        ax1.set_ylabel('F0', color='tomato')
        ax1.tick_params(labelsize='x-small', colors='tomato', bottom=False, labelbottom=False)
        
        ax2 = add_axis(fig, axes[i][0], 1.2)
        ax2.plot(energy, color='darkviolet')
        ax2.set_xlim(0, spectrogram.shape[1])
        ax2.set_ylim(hp.energy_min, hp.energy_max)
        ax2.set_ylabel('Energy', color='darkviolet')
        ax2.yaxis.set_label_position('right')
        ax2.tick_params(labelsize='x-small', colors='darkviolet', bottom=False, labelbottom=False, left=False, labelleft=False, right=True, labelright=True)
       
    plt.savefig(filename, dpi=200)
    plt.clf()

def get_mask_from_lengths(lengths, max_len=None):
    if max_len == None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).byte()

    return mask


def get_WaveGlow():
    waveglow_path = hp.waveglow_path
    wave_glow = torch.load(waveglow_path)['model']
    wave_glow = wave_glow.remove_weightnorm(wave_glow)
    wave_glow.cuda().eval()
    for m in wave_glow.modules():
        if 'Conv' in str(type(m)):
            setattr(m, 'padding_mode', 'zeros')

    return wave_glow


def pad_1D(inputs, PAD=0):

    def pad_data(x, length, PAD):
        x_padded = np.pad(x, (0, length - x.shape[0]),
                          mode='constant',
                          constant_values=PAD)
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
        x_padded = np.pad(x, (0, max_len - np.shape(x)[0]),
                          mode='constant',
                          constant_values=PAD)
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
        max_len = max([input_ele[i].size(0)for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len-batch.size(0)), "constant", 0.0)
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len-batch.size(0)), "constant", 0.0)
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded
    mel = mel[0].cpu().numpy()

    return mel, cemb, D
