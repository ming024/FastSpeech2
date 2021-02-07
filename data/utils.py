import os
import random
import json

import tgt
import torch
import librosa
import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler

import audio as Audio
import hparams as hp


def build_from_path(in_dir, out_dir):
    print("Processing Data ...")
    index = 1
    out = list()
    n_frames = 0
    f0_scaler = StandardScaler()
    energy_scaler = StandardScaler()

    speakers = {}
    for i, speaker in enumerate(os.listdir(in_dir)):
        speakers[speaker] = i
        for wav_name in os.listdir(os.path.join(in_dir, speaker)):
            if ".wav" not in wav_name:
                continue

            basename = wav_name[:-4]
            tg_path = os.path.join(
                out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
            )
            if os.path.exists(tg_path):
                ret = process_utterance(in_dir, out_dir, speaker, basename)
                if ret is None:
                    continue
                else:
                    info, f0, energy, f_max, f_min, e_max, e_min, n = ret
                out.append(info)

            if index % 100 == 0:
                print("Done %d" % index)
            index = index + 1

            if len(f0) > 0 and len(energy) > 0:
                f0_scaler.partial_fit(f0.reshape((-1, 1)))
                energy_scaler.partial_fit(energy.reshape((-1, 1)))

            n_frames += n

    f0_mean = f0_scaler.mean_[0]
    f0_std = f0_scaler.scale_[0]
    energy_mean = energy_scaler.mean_[0]
    energy_std = energy_scaler.scale_[0]

    print("Normalizing Data ...")
    f0_min, f0_max = normalize(os.path.join(out_dir, "f0"), f0_mean, f0_std)
    energy_min, energy_max = normalize(
        os.path.join(out_dir, "energy"), energy_mean, energy_std
    )

    with open(os.path.join(out_dir, "speakers.json"), "w") as f:
        f.write(json.dumps(speakers))

    with open(os.path.join(out_dir, "stat.txt"), "w", encoding="utf-8") as f:
        strs = [
            "Total time: {} hours".format(
                n_frames * hp.hop_length / hp.sampling_rate / 3600
            ),
            "Total frames: {}".format(n_frames),
            "Mean F0: {}".format(f0_mean),
            "Stdev F0: {}".format(f0_std),
            "Min F0: {}".format(f0_min),
            "Max F0: {}".format(f0_max),
            "Min energy: {}".format(energy_min),
            "Max energy: {}".format(energy_max),
        ]
        for s in strs:
            print(s)
            f.write(s + "\n")

    random.shuffle(out)
    out = [r for r in out if r is not None]

    return out


def process_utterance(in_dir, out_dir, speaker, basename):
    wav_path = os.path.join(in_dir, speaker, "{}.wav".format(basename))
    tg_path = os.path.join(out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename))

    # Get alignments
    textgrid = tgt.io.read_textgrid(tg_path)
    phone, duration, start, end = get_alignment(textgrid.get_tier_by_name("phones"))
    text = "{" + " ".join(phone) + "}"
    if start >= end:
        return None

    # Read and trim wav files
    wav, _ = librosa.load(wav_path)
    wav = wav[int(hp.sampling_rate * start) : int(hp.sampling_rate * end)].astype(
        np.float32
    )

    # Compute fundamental frequency
    f0, t = pw.dio(
        wav.astype(np.float64),
        hp.sampling_rate,
        frame_period=hp.hop_length / hp.sampling_rate * 1000,
    )
    f0 = pw.stonemask(wav.astype(np.float64), f0, t, hp.sampling_rate)

    f0 = f0[: sum(duration)]
    if np.all(f0 == 0):
        return None

    # perform linear interpolation
    nonzero_ids = np.where(f0 != 0)[0]
    interp_fn = interp1d(
        nonzero_ids,
        f0[nonzero_ids],
        fill_value=(f0[nonzero_ids[0]], f0[nonzero_ids[-1]]),
        bounds_error=False,
    )
    f0 = interp_fn(np.arange(0, len(f0)))

    # Compute mel-scale spectrogram and energy
    mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav)
    mel_spectrogram = mel_spectrogram.numpy().astype(np.float32)[:, : sum(duration)]
    energy = energy.numpy().astype(np.float32)[: sum(duration)]
    if mel_spectrogram.shape[1] >= hp.max_seq_len:
        return None

    # Phoneme-level average
    pos = 0
    for i, d in enumerate(duration):
        f0[i] = np.mean(f0[pos : pos + d])
        energy[i] = np.mean(energy[pos : pos + d])
        pos += d
    f0 = f0[: len(duration)]
    energy = energy[: len(duration)]

    # Save alignment
    ali_filename = "{}-ali-{}.npy".format(hp.dataset, basename)
    np.save(
        os.path.join(out_dir, "alignment", ali_filename), duration, allow_pickle=False
    )

    # Save fundamental prequency
    f0_filename = "{}-f0-{}.npy".format(hp.dataset, basename)
    np.save(os.path.join(out_dir, "f0", f0_filename), f0, allow_pickle=False)

    # Save energy
    energy_filename = "{}-energy-{}.npy".format(hp.dataset, basename)
    np.save(
        os.path.join(out_dir, "energy", energy_filename), energy, allow_pickle=False
    )

    # Save spectrogram
    mel_filename = "{}-mel-{}.npy".format(hp.dataset, basename)
    np.save(
        os.path.join(out_dir, "mel", mel_filename),
        mel_spectrogram.T,
        allow_pickle=False,
    )

    return (
        "|".join([basename, speaker, text]),
        remove_outlier(f0),
        remove_outlier(energy),
        max(f0),
        min([f for f in f0 if f != 0]),
        max(energy),
        min(energy),
        mel_spectrogram.shape[1],
    )


def get_alignment(tier):
    sil_phones = ["sil", "sp", "spn"]

    phones = []
    durations = []
    durations_real = []
    durations_int = []
    start_time = 0
    end_time = 0
    end_idx = 0
    for t in tier._objects:
        s, e, p = t.start_time, t.end_time, t.text

        # Trimming leading silences
        if phones == []:
            if p in sil_phones:
                continue
            else:
                start_time = s
        if p not in sil_phones:
            phones.append(p)
            end_time = e
            end_idx = len(phones)
        else:
            phones.append(p)
        durations.append(
            int(
                np.round(e * hp.sampling_rate / hp.hop_length)
                - np.round(s * hp.sampling_rate / hp.hop_length)
            )
        )

    # Trimming tailing silences
    phones = phones[:end_idx]
    durations = durations[:end_idx]

    return phones, durations, start_time, end_time


def remove_outlier(values):
    p25 = np.percentile(values, 25)
    p75 = np.percentile(values, 75)
    lower = p25 - 1.5 * (p75 - p25)
    upper = p75 + 1.5 * (p75 - p25)
    normal_indices = np.logical_and(values > lower, values < upper)
    return values[normal_indices]


def normalize(in_dir, mean, std):
    max_value = -100
    min_value = 100
    for filename in os.listdir(in_dir):
        filename = os.path.join(in_dir, filename)
        values = (np.load(filename) - mean) / std
        np.save(filename, values, allow_pickle=False)

        max_value = max(max_value, max(values))
        min_value = min(min_value, min(values))

    return min_value, max_value
