import json
import math
import os
import csv
from string import ascii_lowercase

import numpy as np
from torch.utils.data import Dataset

from text import text_to_sequence
from utils.tools import pad_1D, pad_2D, pad_SPE_D
from text.cleaners import CLEANERS


class Dataset(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False
    ):
        self.use_spe_features = preprocess_config["preprocessing"]["text"][
            "use_spe_features"
        ]
        self.spe_feature_dim = preprocess_config["preprocessing"]["text"][
            "spe_feature_dim"
        ]
        self.phones = {}
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]
        if (
            "use_moh_reader" in preprocess_config["preprocessing"]
            and preprocess_config["preprocessing"]["use_moh_reader"]
        ):
            (
                self.basename,
                self.language,
                self.speaker,
                self.text,
                self.raw_text,
            ) = self.process_moh_meta(filename)
        elif (
            "use_git_reader" in preprocess_config["preprocessing"]
            and preprocess_config["preprocessing"]["use_git_reader"]
        ):
            (
                self.basename,
                self.language,
                self.speaker,
                self.text,
                self.raw_text,
            ) = self.process_git_meta(filename)
        else:
            (
                self.basename,
                self.language,
                self.speaker,
                self.text,
                self.raw_text,
            ) = self.process_meta(filename)
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        with open(os.path.join(self.preprocessed_path, "languages.json")) as f:
            self.language_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        language = self.language[idx]
        language_id = self.language_map[language]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        if idx not in self.phones:
            if self.use_spe_features:
                feat_path = os.path.join(
                    self.preprocessed_path,
                    "feature",
                    "{}-{}-feat-{}.npy".format(language, speaker, basename),
                )
                self.phones[idx] = np.load(feat_path)
            else:
                self.phones[idx] = np.array(
                    text_to_sequence(self.text[idx], self.use_spe_features, language)
                )
        phone = self.phones[idx]
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-{}-mel-{}.npy".format(language, speaker, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-{}-pitch-{}.npy".format(language, speaker, basename),
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-{}-energy-{}.npy".format(language, speaker, basename),
        )
        energy = np.load(energy_path)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-{}-duration-{}.npy".format(language, speaker, basename),
        )
        duration = np.load(duration_path)

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "language": language_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
        }

        return sample

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            language = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, l, s, t, r = line.strip("\n").split("|")
                name.append(n)
                language.append(l)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, language, speaker, text, raw_text

    def process_moh_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename),
            "r",
            newline="",
            encoding="utf8",
        ) as f:
            reader = csv.reader(
                f, delimiter="|", quoting=csv.QUOTE_NONE, escapechar="\\"
            )
            filepaths_and_text = [[x[0], "harvey", x[4], x[2]] for x in reader]
        name = [x[0] for x in filepaths_and_text]

        speaker = [x[1] for x in filepaths_and_text]
        text = [x[2] for x in filepaths_and_text]
        raw_text = [x[3] for x in filepaths_and_text]
        return name, speaker, text, raw_text

    def process_git_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename),
            "r",
            newline="",
            encoding="utf8",
        ) as f:
            reader = csv.reader(
                f, delimiter="|", quoting=csv.QUOTE_NONE, escapechar="\\"
            )
            filepaths_and_text = [[x[0], x[1], x[2], x[3], x[4]] for x in reader]
        name = [x[0] for x in filepaths_and_text]
        language = [x[1] for x in filepaths_and_text]
        speaker = [x[2] for x in filepaths_and_text]
        text = [x[3] for x in filepaths_and_text]
        raw_text = [x[4] for x in filepaths_and_text]
        return name, language, speaker, text, raw_text

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        languages = [data[idx]["language"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])
        languages = np.array(languages)
        speakers = np.array(speakers)
        if self.use_spe_features:
            texts = pad_SPE_D(texts, self.spe_feature_dim)
        else:
            texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)

        return (
            ids,
            raw_texts,
            languages,
            speakers,
            texts,
            text_lens,
            max(text_lens),
            mels,
            mel_lens,
            max(mel_lens),
            pitches,
            energies,
            durations,
        )

    def collate_fn(self, data):
        data_size = len(data)
        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output


class TextDataset(Dataset):
    # AP: This is only used in batch inference...
    def __init__(self, filepath, preprocess_config):
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.use_spe_features = preprocess_config["preprocessing"]["text"][
            "use_spe_features"
        ]
        self.spe_feature_dim = preprocess_config["preprocessing"]["text"][
            "spe_feature_dim"
        ]
        self.phones = {}
        if (
            "use_moh_reader" in preprocess_config["preprocessing"]
            and preprocess_config["preprocessing"]["use_moh_reader"]
        ):
            (
                self.basename,
                self.language,
                self.speaker,
                self.text,
                self.raw_text,
            ) = self.process_moh_meta(filepath)
        elif (
            "use_git_reader" in preprocess_config["preprocessing"]
            and preprocess_config["preprocessing"]["use_git_reader"]
        ):
            (
                self.basename,
                self.language,
                self.speaker,
                self.text,
                self.raw_text,
            ) = self.process_git_meta(filepath)
        else:
            (
                self.basename,
                self.language,
                self.speaker,
                self.text,
                self.raw_text,
            ) = self.process_meta(filepath)
        with open(
            os.path.join(
                preprocess_config["path"]["preprocessed_path"], "speakers.json"
            )
        ) as f:
            self.speaker_map = json.load(f)
        with open(
            os.path.join(
                preprocess_config["path"]["preprocessed_path"], "languages.json"
            )
        ) as f:
            self.language_map = json.load(f)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        language = self.language[idx]
        language_id = self.language_map[language]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        if idx not in self.phones:
            self.phones[idx] = np.array(
                text_to_sequence(self.text[idx], self.use_spe_features, language)
            )
        phone = self.phones[idx]
        return (basename, language_id, speaker_id, phone, raw_text)

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            language = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, l, s, t, r = line.strip("\n").split("|")
                name.append(n)
                language.append(l)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, language, speaker, text, raw_text

    def collate_fn(self, data):
        ids = [d[0] for d in data]
        languages = np.array([d[1] for d in data])
        speakers = np.array([d[2] for d in data])
        texts = [d[3] for d in data]
        raw_texts = [d[4] for d in data]
        text_lens = np.array([text.shape[0] for text in texts])

        if self.use_spe_features:
            texts = pad_SPE_D(texts, self.spe_feature_dim)
        else:
            texts = pad_1D(texts)

        return ids, raw_texts, languages, speakers, texts, text_lens, max(text_lens)


if __name__ == "__main__":
    # Test
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from utils.utils import to_device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess_config = yaml.load(
        open("./config/LJSpeech/preprocess.yaml", "r"), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open("./config/LJSpeech/train.yaml", "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(
        open("./config/LJSpeech/model.yaml", "r"), Loader=yaml.FullLoader
    )

    use_energy = model_config["variance_predictor"]["use_energy_predictor"]

    if "train_file" in train_config["path"] and train_config["path"]["train_file"]:
        train_file = train_config["path"]["train_file"]
    else:
        train_file = "train.txt"

    if "val_file" in train_config["path"] and train_config["path"]["val_file"]:
        val_file = train_config["path"]["val_file"]
    else:
        val_file = "val.txt"

    train_dataset = Dataset(
        train_file, preprocess_config, train_config, sort=True, drop_last=True
    )
    val_dataset = Dataset(
        val_file, preprocess_config, train_config, sort=False, drop_last=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["optimizer"]["batch_size"] * 4,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["optimizer"]["batch_size"],
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )

    n_batch = 0
    for batchs in train_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print(
        "Training set  with size {} is composed of {} batches.".format(
            len(train_dataset), n_batch
        )
    )

    n_batch = 0
    for batchs in val_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print(
        "Validation set  with size {} is composed of {} batches.".format(
            len(val_dataset), n_batch
        )
    )
