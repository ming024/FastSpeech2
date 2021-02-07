import os

import librosa
import numpy as np
from scipy.io import wavfile

import hparams as hp


def prepare_align(in_dir):
    folders = [
        "MST-Originbeat-S1-female-5000",
        "MST-Originbeat-S2-male-5000",
        "TSV-Track1-S1-female-Anchor-100",
        "TSV-Track1-S2-male-Sales-100",
        "TSV-Track2-S1-female-5",
        "TSV-Track2-S2-male-5",
        "TST-Track1-S3-female-Chat-100",
        "TST-Track1-S4-male-Game-100",
        "TST-Track1-S5-male-Story-100",
        "TST-Track2-S3-female-5",
        "TST-Track2-S4-male-5",
        "TST-Track2-S5-male-5",
    ]
    speakers = [
        "MST_S1",
        "MST_S2",
        "TSV_T1_S1",
        "TSV_T1_S2",
        "TSV_T2_S1",
        "TSV_T2_S2",
        "TST_T1_S3",
        "TST_T1_S4",
        "TST_T1_S5",
        "TST_T2_S3",
        "TST_T2_S4",
        "TST_T2_S5",
    ]
    for folder, speaker in zip(folders, speakers):
        with open(os.path.join(in_dir, folder, folder + ".txt"), encoding="utf-8") as f:
            while True:
                try:
                    wav_name = f.readline().strip("\n")
                    next(f)
                    text = f.readline().strip("\n").replace("[]", "").replace("  ", " ")
                    next(f)

                    wav_path = os.path.join(in_dir, folder, "wavs", wav_name + ".wav")
                    if os.path.exists(wav_path):
                        os.makedirs(os.path.join(hp.raw_path, speaker), exist_ok=True)
                        wav, _ = librosa.load(wav_path, hp.sampling_rate)
                        wav = wav / max(abs(wav)) * hp.max_wav_value
                        wavfile.write(
                            os.path.join(
                                hp.raw_path,
                                speaker,
                                "{}_{}.wav".format(speaker, wav_name),
                            ),
                            hp.sampling_rate,
                            wav.astype(np.int16),
                        )
                        with open(
                            os.path.join(
                                hp.raw_path,
                                speaker,
                                "{}_{}.lab".format(speaker, wav_name),
                            ),
                            "w",
                        ) as f1:
                            f1.write(text)

                    if speaker[:3] == "MST":
                        next(f)
                    next(f)

                except:
                    break