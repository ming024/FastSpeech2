import os

import hparams as hp
from data.utils import build_from_path


def write_metadata(train, out_dir):
    with open(os.path.join(out_dir, "train.txt"), "w", encoding="utf-8") as f:
        for m in train:
            f.write(m + "\n")


def main():
    in_dir = hp.raw_path
    out_dir = hp.preprocessed_path
    mel_out_dir = os.path.join(out_dir, "mel")
    if not os.path.exists(mel_out_dir):
        os.makedirs(mel_out_dir, exist_ok=True)
    ali_out_dir = os.path.join(out_dir, "alignment")
    if not os.path.exists(ali_out_dir):
        os.makedirs(ali_out_dir, exist_ok=True)
    f0_out_dir = os.path.join(out_dir, "f0")
    if not os.path.exists(f0_out_dir):
        os.makedirs(f0_out_dir, exist_ok=True)
    energy_out_dir = os.path.join(out_dir, "energy")
    if not os.path.exists(energy_out_dir):
        os.makedirs(energy_out_dir, exist_ok=True)

    train = build_from_path(in_dir, out_dir)
    write_metadata(train, out_dir)


if __name__ == "__main__":
    main()
