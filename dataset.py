import math
import os
import json

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

import hparams as hp
import audio as Audio
from utils import pad_1D, pad_2D
from text import text_to_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dataset(Dataset):
    def __init__(self, filename="train.txt", sort=True):
        self.basename, self.speaker, self.text = self.process_meta(
            os.path.join(hp.preprocessed_path, filename)
        )
        self.sort = sort
        with open(os.path.join(hp.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        while True:
            try:
                basename = self.basename[idx]
                speaker_id = self.speaker_map[self.speaker[idx]]
                phone = np.array(text_to_sequence(self.text[idx]))
                mel_path = os.path.join(
                    hp.preprocessed_path,
                    "mel",
                    "{}-mel-{}.npy".format(hp.dataset, basename),
                )
                mel_target = np.load(mel_path)
                # print("mel loaded")
                D_path = os.path.join(
                    hp.preprocessed_path,
                    "alignment",
                    "{}-ali-{}.npy".format(hp.dataset, basename),
                )
                D = np.load(D_path)
                # print("alignment loaded")
                f0_path = os.path.join(
                    hp.preprocessed_path,
                    "f0",
                    "{}-f0-{}.npy".format(hp.dataset, basename),
                )
                f0 = np.load(f0_path)
                # print("f0 loaded")
                energy_path = os.path.join(
                    hp.preprocessed_path,
                    "energy",
                    "{}-energy-{}.npy".format(hp.dataset, basename),
                )
                energy = np.load(energy_path)
                # print("energy loaded")

                # x_vec_path = os.path.join(
                #     hp.preprocessed_path,
                #     "x_vec",
                #     "{}-xvector-{}.npy".format(hp.dataset, basename),
                # )
                # x_vec = np.load(x_vec_path)
                x_vec = np.empty(0)
                # d_vec_path = os.path.join(
                #     hp.preprocessed_path,
                #     "d_vec",
                #     "{}-dvector-{}.npy".format(hp.dataset, basename),
                # )
                # d_vec = np.load(d_vec_path)
                d_vec = np.empty(0)
                adain_emb_path = os.path.join(
                    hp.preprocessed_path,
                    "adain",
                    "{}-adain-{}.npy".format(hp.dataset, basename),
                )
                adain_emb = np.load(adain_emb_path)
                # print("adain loaded")
                sample = {
                    "id": basename,
                    "speaker": speaker_id,
                    "text": phone,
                    "mel_target": mel_target,
                    "D": D,
                    "f0": f0,
                    "energy": energy,
                    "x_vec": x_vec,
                    "d_vec": d_vec,
                    "adain": adain_emb,
                }
                break
            except FileNotFoundError as e:
                idx = (idx + 1) % self.__len__()

        return sample

    def process_meta(self, meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            text = []
            speaker = []
            name = []
            for line in f.readlines():
                n, s, t = line.strip("\n").split("|")
                # if "TSV_T2" not in s:
                name.append(n)
                speaker.append(s)
                text.append(t)
            return name, speaker, text

    def reprocess(self, batch, cut_list):
        ids = [batch[ind]["id"] for ind in cut_list]
        speakers = [batch[ind]["speaker"] for ind in cut_list]
        texts = [batch[ind]["text"] for ind in cut_list]
        mel_targets = [batch[ind]["mel_target"] for ind in cut_list]
        Ds = [batch[ind]["D"] for ind in cut_list]
        f0s = [batch[ind]["f0"] for ind in cut_list]
        energies = [batch[ind]["energy"] for ind in cut_list]
        x_vec = np.array([batch[ind]["x_vec"] for ind in cut_list])
        d_vec = np.array([batch[ind]["d_vec"] for ind in cut_list])
        adain = np.array([batch[ind]["adain"] for ind in cut_list])
        for text, D, id_ in zip(texts, Ds, ids):
            if len(text) != len(D):
                print(text, text.shape, D, D.shape, id_)
        length_text = np.array(list())
        for text in texts:
            length_text = np.append(length_text, text.shape[0])

        length_mel = np.array(list())
        for mel in mel_targets:
            length_mel = np.append(length_mel, mel.shape[0])

        speakers = np.array(speakers)
        texts = pad_1D(texts)
        mel_targets = pad_2D(mel_targets)
        Ds = pad_1D(Ds)
        f0s = pad_1D(f0s)
        energies = pad_1D(energies)
        log_Ds = np.log(Ds + hp.log_offset)

        out = {
            "id": ids,
            "speaker": speakers,
            "text": texts,
            "mel_target": mel_targets,
            "D": Ds,
            "log_D": log_Ds,
            "f0": f0s,
            "energy": energies,
            "src_len": length_text,
            "mel_len": length_mel,
            "x_vec": x_vec,
            "d_vec": d_vec,
            "adain": adain,
        }

        return out

    def collate_fn(self, batch):
        len_arr = np.array([d["text"].shape[0] for d in batch])
        index_arr = np.argsort(-len_arr)
        batchsize = len(batch)
        real_batchsize = int(math.sqrt(batchsize))

        cut_list = list()
        for i in range(real_batchsize):
            if self.sort:
                cut_list.append(
                    index_arr[i * real_batchsize : (i + 1) * real_batchsize]
                )
            else:
                cut_list.append(np.arange(i * real_batchsize, (i + 1) * real_batchsize))

        output = list()
        for i in range(real_batchsize):
            output.append(self.reprocess(batch, cut_list[i]))

        return output


if __name__ == "__main__":
    # Test
    dataset = Dataset("train.txt")
    training_loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        drop_last=True,
        num_workers=0,
    )

    cnt = 0
    for i, batchs in enumerate(training_loader):
        for j, data_of_batch in enumerate(batchs):
            mel_target = (
                torch.from_numpy(data_of_batch["mel_target"]).float().to(device)
            )
            D = torch.from_numpy(data_of_batch["D"]).int().to(device)
            if mel_target.shape[1] == D.sum().item():
                cnt += 1

    print(cnt, len(dataset))
