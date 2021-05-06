import argparse
import os
from data.utils import build_from_path
from hparams import sampling_rate
import shutil


def move_textgrids(path, textgrid):
    for spk in os.listdir(textgrid):
        spk_dir = os.path.join(textgrid, spk)
        for fn in os.listdir(spk_dir):
            shutil.copy(os.path.join(spk_dir, fn), os.path.join(path, spk, fn))
            # shutil.move(os.path.join(spk_dir, fn), os.path.join(path, spk, fn))


def train_test_split(dataset, target_dataset):
    for spk in os.listdir(dataset):
        train = os.path.join(target_dataset, "train", spk)
        dev = os.path.join(target_dataset, "dev", spk)

        os.makedirs(train, exist_ok=True)
        os.makedirs(dev, exist_ok=True)

        for fn in os.listdir(os.path.join(dataset, spk)):
            full_fn = os.path.join(dataset, spk, fn)
            if "_mic" in full_fn:
                file_data = "_".join(fn.split("_")[:2])
            else:
                file_data = fn.split(".")[0]
            is_dev = abs(hash(file_data)) % 10 == 0
            if not is_dev:
                shutil.copy(full_fn, os.path.join(train, fn))
            else:
                shutil.copy(full_fn, os.path.join(dev, fn))


def process(path, target_path):
    suffix = "preprocessed"
    path_preprocessed = os.path.join(target_path, suffix)
    os.makedirs(os.path.join(path_preprocessed, "mel"), exist_ok=True)
    os.makedirs(os.path.join(path_preprocessed, "alignment"), exist_ok=True)
    os.makedirs(os.path.join(path_preprocessed, "f0"), exist_ok=True)
    os.makedirs(os.path.join(path_preprocessed, "energy"), exist_ok=True)
    build_from_path(path, path_preprocessed, sampling_rate=sampling_rate)


def main(path, target_path, textgrid):
    if textgrid:
        print("Moving textgrids")
        move_textgrids(path, textgrid)
    # print("Splitting to train/dev")
    # train_test_split(path, target_path)
    print("Preprocessing")
    process(path, target_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--target_path", type=str, required=True)
    parser.add_argument("--textgrid", type=str, required=False)
    args = parser.parse_args()
    main(args.path, args.target_path, args.textgrid)
