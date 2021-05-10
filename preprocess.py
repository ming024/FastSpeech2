import argparse
import os
from data.utils import build_from_path
from hparams import sampling_rate
import shutil
from adain import gen_adain


def move_textgrids(path, textgrid):
    for spk in os.listdir(textgrid):
        spk_dir = os.path.join(textgrid, spk)
        for fn in os.listdir(spk_dir):
            shutil.copy(os.path.join(spk_dir, fn), os.path.join(path, spk, fn.replace('-', '_')))


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
    os.makedirs(os.path.join(target_path, "mel"), exist_ok=True)
    os.makedirs(os.path.join(target_path, "alignment"), exist_ok=True)
    os.makedirs(os.path.join(target_path, "f0"), exist_ok=True)
    os.makedirs(os.path.join(target_path, "energy"), exist_ok=True)
    build_from_path(path, target_path, sampling_rate=sampling_rate)


def main(path, target_path, textgrid, need_adains):
    if textgrid:
        print("Moving textgrids")
        move_textgrids(path, textgrid)
    print("Splitting to train/dev")
    train_test_split(path, target_path)
    print("Preprocessing")
    process(target_path, os.path.join(target_path, "preprocessed"))
    if need_adains:
        print("Creating adain embeddings")
        gen_adain(path + '/train', target_path + '/adain')
        gen_adain(path + '/dev', target_path + '/adain')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True,
                        help="source dataset: train/dev->speaker_id->wav+lab+(optional)textgrid")
    parser.add_argument("--target_path", type=str, required=True, help="path to save preprocessed data")
    parser.add_argument("--textgrid", type=str, required=False, help="directory with textgrid files")
    parser.add_argument("--adain", type=bool, required=False, help="create adain embeddings")
    args = parser.parse_args()
    main(args.path, args.target_path, args.textgrid, args.adain)
