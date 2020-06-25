import numpy as np
import os
import random
import re
import tgt
from scipy.io.wavfile import read
import pyworld as pw
import torch
import audio as Audio
from utils import get_alignment
import hparams as hp

def prepare_align(in_dir):
    with open(os.path.join(in_dir, 'prompts.gui'), encoding='utf-8') as f:
        for line in f:
            basename = line.strip('\n')
            wav_path = os.path.join(in_dir, 'wavn', '{}.wav'.format(basename))
            if os.path.exists(wav_path):
                text = re.sub(' +', ' ', re.sub(r'[#@|]', '', next(f).strip())).strip(' ')
                text = re.sub(r'\s([?.!":,-;\'\"](?:\s|$))', r'\1', text)
            
                with open(os.path.join(in_dir, 'wavn', '{}.txt'.format(basename)), 'w') as f1:
                    f1.write(text)

def build_from_path(in_dir, out_dir):
    index = 1
    out = list()
    f0_max = energy_max = 0
    f0_min = energy_min = 1000000
    n_frames = 0

    with open(os.path.join(in_dir, 'prompts.gui'), encoding='utf-8') as f:
        for line in f:
            basename = line.strip('\n')
            tg_path = os.path.join(out_dir, 'TextGrid', '{}.TextGrid'.format(basename))
            if os.path.exists(tg_path):
                text = re.sub(' +', ' ', re.sub(r'[#@|]', '', next(f).strip())).strip(' ')
                text = re.sub(r'\s([?.!":,-;\'\"](?:\s|$))', r'\1', text)
            
                ret = process_utterance(in_dir, out_dir, basename)
                if ret is None:
                    continue
                else:
                    info, f_max, f_min, e_max, e_min, n = ret
                out.append(info)

                if index % 100 == 0:
                    print("Done %d" % index)
                index = index + 1
                
                f0_max = max(f0_max, f_max)
                f0_min = min(f0_min, f_min)
                energy_max = max(energy_max, e_max)
                energy_min = min(energy_min, e_min)
                n_frames += n

    with open(os.path.join(out_dir, 'stat.txt'), 'w', encoding='utf-8') as f:
        strs = ['Total time: {} hours'.format(n_frames*hp.hop_length/hp.sampling_rate/3600),
                'Total frames: {}'.format(n_frames),
                'Min F0: {}'.format(f0_min),
                'Max F0: {}'.format(f0_max),
                'Min energy: {}'.format(energy_min),
                'Max energy: {}'.format(energy_max)]
        for s in strs:
            print(s)
            f.write(s+'\n')
    
    random.shuffle(out)
    out = [r for r in out if r is not None]

    return out[hp.eval_size:], out[:hp.eval_size]

def process_utterance(in_dir, out_dir, basename):
    wav_path = os.path.join(in_dir, 'wavn', '{}.wav'.format(basename))
    tg_path = os.path.join(out_dir, 'TextGrid', '{}.TextGrid'.format(basename)) 
    
    # Get alignments
    textgrid = tgt.io.read_textgrid(tg_path)
    phone, duration, start, end = get_alignment(textgrid.get_tier_by_name('phones'))
    text = '{'+ ' '.join(phone) + '}'
    text = text.replace(' $ ', '} {') # $ represents silent phones
    if start >= end:
        return None
    
    # Read and trim wav files
    _, wav = read(wav_path)
    wav = wav[int(hp.sampling_rate*start):int(hp.sampling_rate*end)].astype(np.float32)
    
    # Compute fundamental frequency
    f0, _ = pw.dio(wav.astype(np.float64), hp.sampling_rate, frame_period=hp.hop_length/hp.sampling_rate*1000)
    f0 = f0[:sum(duration)]

    # Compute mel-scale spectrogram
    mel_spectrogram = Audio.tools.get_mel_from_wav(torch.FloatTensor(wav)).numpy().astype(np.float32)
    mel_spectrogram = mel_spectrogram[:, :sum(duration)]
    if mel_spectrogram.shape[1] >= hp.max_seq_len:
        return None

    # Compute energy
    energy = np.linalg.norm(mel_spectrogram, axis=0)

    # Save alignment
    ali_filename = '{}-ali-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'alignment', ali_filename), duration, allow_pickle=False)

    # Save fundamental prequency
    f0_filename = '{}-f0-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'f0', f0_filename), f0, allow_pickle=False)

    # Save energy
    energy_filename = '{}-energy-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'energy', energy_filename), energy, allow_pickle=False)

    # Save spectrogram
    mel_filename = '{}-mel-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'mel', mel_filename), mel_spectrogram.T, allow_pickle=False)
    
    return '|'.join([basename, text]), max(f0), min([f for f in f0 if f != 0]), max(energy), min(energy), mel_spectrogram.shape[1]
