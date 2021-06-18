import os

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm


def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    # max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    for dataset in ["train", "test"]:
        print("Processing {}ing set...".format(dataset))
        with open(os.path.join(in_dir, dataset, "content.txt"), encoding="utf-8") as f:
            for line in tqdm(f):
                wav_name, text = line.strip("\n").split("\t")
                speaker = wav_name[:7]
                text = text.split(" ")[1::2]
                wav_path = os.path.join(in_dir, dataset, "wav", speaker, wav_name)
                if os.path.exists(wav_path):
                    os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
                    wav, _ = librosa.load(wav_path, sampling_rate)
                    wav = wav / max(abs(wav))  # * max_wav_value
                    sf.write(
                        os.path.join(out_dir, speaker, wav_name),
                        wav,
                        sampling_rate,
                        subtype='PCM_16'
                    )
                    with open(
                        os.path.join(out_dir, speaker, "{}.lab".format(wav_name[:11])),
                        "w",
                    ) as f1:
                        f1.write(" ".join(text))