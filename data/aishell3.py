import os

import librosa
import numpy as np
from scipy.io import wavfile

import hparams as hp
from tqdm.auto import tqdm

from torch.utils.data import DataLoader
from dataset import PrepareAlignDataset

def prepare_align(in_dir):
    # for dataset in ["train", "test"]:
    #     with open(os.path.join(in_dir, dataset, "content.txt"), encoding="utf-8") as f:
    #         for line in f:
    #             wav_name, text = line.strip("\n").split("\t")
    #             speaker = wav_name[:7]
    #             text = text.split(" ")[1::2]
    #             wav_path = os.path.join(in_dir, dataset, "wav", speaker, wav_name)
    #             if os.path.exists(wav_path):
    #                 os.makedirs(os.path.join(hp.raw_path, speaker), exist_ok=True)
    #                 wav, _ = librosa.load(wav_path, hp.sampling_rate)
    #                 wav = wav / max(abs(wav)) * hp.max_wav_value
    #                 wavfile.write(
    #                     os.path.join(hp.raw_path, speaker, wav_name),
    #                     hp.sampling_rate,
    #                     wav.astype(np.int16),
    #                 )
    #                 with open(
    #                     os.path.join(
    #                         hp.raw_path, speaker, "{}.lab".format(wav_name[:11])
    #                     ),
    #                     "w",
    #                 ) as f1:
    #                     f1.write(" ".join(text))

    # For Training
    # dataset = PrepareAlignDataset(in_dir, audio_ext=".wav", text_ext=".normalized.txt")
    dataset = PrepareAlignDataset(in_dir, audio_ext=".wav", text_ext=".lab")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda b: zip(*b))

    for wav_path, wav, text_path, text in tqdm(dataloader):
        wav_path, wav, text_path, text = wav_path[0], wav[0], text_path[0], text[0]
        wavfile.write(wav_path, hp.sampling_rate, wav)
        with open(text_path, "w") as f1:
            f1.write(text)

    # # For validation
    # for speaker in tqdm(os.listdir(in_dir)):
    #     for chapter in os.listdir(os.path.join(in_dir, speaker)):
    #         for file_name in os.listdir(os.path.join(in_dir, speaker, chapter)):
    #             if file_name[-4:] != ".wav":
    #                 continue
    #             base_name = file_name[:-4]
    #             wav_path = os.path.join(
    #                 in_dir, speaker, chapter, "{}.wav".format(base_name)
    #             )

    #             os.makedirs(os.path.join(hp.raw_path, speaker), exist_ok=True)
    #             wav, _ = librosa.load(wav_path, hp.sampling_rate)
    #             wav = wav / max(abs(wav)) * hp.max_wav_value
    #             wavfile.write(
    #                 os.path.join(hp.raw_path, speaker, "{}.wav".format(base_name)),
    #                 hp.sampling_rate,
    #                 wav.astype(np.int16),
    #             )