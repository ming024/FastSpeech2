import argparse
import os
import re
import time
from string import punctuation

import torch
import torch.nn as nn
import numpy as np
from pypinyin import pinyin, Style
from g2p_en import G2p

import utils
import hparams as hp
import audio as Audio
from text import text_to_sequence
from model.fastspeech2 import FastSpeech2
from plot.utils import plot_mel



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_english(text):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon("phoneme_dict")

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones
        )
    )

    return np.array(sequence)


def read_source(source_path):
    ids = []
    sequences = []
    with open(source_path, "r") as f:
        for line in f:
            # id_, sequence, sentence = line.strip("\n").split("|")
            # id_, sentence = line.strip("\n").split("|")
            id_, speaker, sequence = line.strip("\n").split("|")
            ids.append(id_)

            sequences.append(np.array(text_to_sequence(sequence)))
            # sequences.append(preprocess_english(sentence))
    return ids, sequences



def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def get_FastSpeech2(step):
    checkpoint_path = os.path.join(
        hp.checkpoint_path, "checkpoint_{}.pth.tar".format(step)
    )

    ckpt = torch.load(checkpoint_path)["model"]

    # speaker_num = len(utils.get_speaker_to_id())
    # print(ckpt.keys())
    speaker_num = ckpt['module.speaker_embedding.weight'].shape[0]
    print("Num speakers:", speaker_num)

    model = nn.DataParallel(FastSpeech2(speaker_num))
    model.load_state_dict(ckpt, strict=False)
    model.requires_grad = False
    model.eval()
    return model


def synthesize(
    model,
    vocoder,
    d_vec,
    x_vec,
    adain,
    speaker,
    gst,
    texts,
    file_ids,
    mel_len,
    Ds,
    f0s,
    energies,
    prefix="",
):
    src_len = torch.from_numpy(np.array([len(t) for t in texts])).to(device)
    texts = torch.from_numpy(utils.pad_1D(texts)).to(device)
    Ds = torch.from_numpy(utils.pad_1D(Ds)).to(device) if Ds is not None else None
    f0s = torch.from_numpy(utils.pad_1D(f0s)).to(device) if f0s is not None else None
    energies = torch.from_numpy(utils.pad_1D(energies)).to(device) if energies is not None else None
    mel_len = torch.from_numpy(mel_len).long().to(device) if mel_len is not None else None


    d_vec = (
        torch.from_numpy(np.array(d_vec))
        .to(device)
        .unsqueeze(0)
        .expand(len(file_ids), -1)
        if d_vec is not None
        else None
    )
    x_vec = (
        torch.from_numpy(np.array(x_vec))
        .to(device)
        .unsqueeze(0)
        .expand(len(file_ids), -1)
        if x_vec is not None
        else None
    )
    adain = (
        torch.from_numpy(np.array(adain))
        .to(device)
        .unsqueeze(0)
        .expand(len(file_ids), -1)
        if adain is not None
        else None
    )
    speakers = (
        torch.from_numpy(np.array([speaker])).to(device).expand(len(file_ids))
        if speaker is not None
        else None
    )
    gst = (
        torch.from_numpy(np.array(gst))
        .to(device)
        .unsqueeze(0)
        .expand(len(file_ids), -1)
        if gst is not None
        else None
    )

    (
        mel,
        mel_postnet,
        log_duration_output,
        duration_output,
        f0_output,
        energy_output,
        _,
        _,
        mel_len,
    ) = model(
        texts,
        src_len,
        mel_len=mel_len,
        d_target=Ds,
        p_target=f0s,
        e_target=energies,
        max_src_len=torch.max(src_len).item(),
        d_vec=d_vec,
        x_vec=x_vec,
        adain=adain,
        speaker=speakers,
        use_gst=args.gst,
        gst=gst,
    )

    if not os.path.exists(hp.test_path):
        os.makedirs(hp.test_path)

    utils.vocoder_infer(
        mel_postnet.transpose(1, 2),
        vocoder,
        [
            os.path.join(hp.test_path, "{}_{}.wav".format(prefix, file_id))
            for file_id in file_ids
        ],
        mel_len * hp.hop_length,
    )

    for i in range(len(texts)):
        file_id = file_ids[i]
        src_length = src_len[i]
        mel_length = mel_len[i]
        mel_postnet_ = (
            mel_postnet[i, :mel_length].transpose(0, 1).detach().cpu().numpy()
        )
        f0_output_ = f0_output[i, :src_length].detach().cpu().numpy()
        energy_output_ = energy_output[i, :src_length].detach().cpu().numpy()
        duration_output_ = (
            duration_output[i, :src_length].detach().cpu().numpy().astype(np.int)
        )

        np.save(
            os.path.join(hp.test_path, "{}_{}.npy".format(prefix, file_id)),
            mel_postnet_.T,
        )

        plot_mel(
            [(mel_postnet_, f0_output_, energy_output_, duration_output_)],
            ["Synthesized Spectrogram"],
            filename=os.path.join(hp.test_path, "{}_{}.png".format(prefix, file_id)),
        )


if __name__ == "__main__":
    # Test
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=500000)
    parser.add_argument(
        "--speaker",
        type=str,
        # choices=[
        #     # "TST_T1_S3",
        #     # "TST_T1_S4",
        #     # "TST_T1_S5",
        #     # "TST_T2_S3",
        #     # "TST_T2_S4",
        #     # "TST_T2_S5",
        #     # 1970 2691 3214 4859 5703 6454
        #     "1970",
        #     "2691",
        #     "3214",
        #     "4859",
        #     "5703",
        #     "6454",
        # ],
    )
    parser.add_argument("--source", type=str)
    parser.add_argument("--d_vec", action="store_true")
    parser.add_argument("--x_vec", action="store_true")
    parser.add_argument("--adain", action="store_true")
    parser.add_argument("--speaker_emb", action="store_true")
    parser.add_argument("--gst", action="store_true")

    args = parser.parse_args()

    src_prep_path = f"./preprocessed_data/{hp.target_dataset}"
    file_ids, texts = read_source(args.source)
    Ds = [
        np.load(os.path.join(src_prep_path, "alignment", f"{hp.target_dataset}-ali-{id_}.npy"))
        for id_ in file_ids
    ]
    # f0s = [
    #     np.load(os.path.join(src_prep_path, "f0", f"{hp.target_dataset}-f0-{id_}.npy"))
    #     for id_ in file_ids
    # ]
    f0s = None
    energies = [
        np.load(os.path.join(src_prep_path, "energy", f"{hp.target_dataset}-energy-{id_}.npy"))
        for id_ in file_ids
    ]
    # energies = None
    source_mels = [
        np.load(os.path.join(src_prep_path, "mel", f"{hp.target_dataset}-mel-{id_}.npy"))
        for id_ in file_ids
    ]

    mel_lens = np.array(list())
    for mel in source_mels:
        mel_lens = np.append(mel_lens, mel.shape[0])

    speaker_to_track = {
        # "TST_T1_S3": "Track1_b",
        # "TST_T1_S4": "Track1_b",
        # "TST_T1_S5": "Track1_b",
        # "TST_T2_S3": "Track2_b",
        # "TST_T2_S4": "Track2_b",
        # "TST_T2_S5": "Track2_b",
        "1970": "Track1_b",
        "2691": "Track1_b",
        "3214": "Track1_b",
        "4859": "Track1_b",
        "5703": "Track1_b",
        "6454": "Track1_b",
    }
    speaker_to_id = utils.get_speaker_to_id()

    # prefix = "{}_{}".format(speaker_to_track[args.speaker], args.speaker)
    prefix = "{}_{}".format('Valid_1shot', args.speaker)

    # Get averaged speaker embedding
    mel_path = os.path.join(hp.preprocessed_path, "mel")
    reference_mels = [
        np.load(os.path.join(mel_path, filename))
        for filename in os.listdir(mel_path)
        if args.speaker in filename
    ]

    d_vec = (
        np.mean(utils.get_pretrained_embedding(args.speaker, "d_vec"), 0)
        if args.d_vec
        else None
    )
    x_vec = (
        np.mean(utils.get_pretrained_embedding(args.speaker, "x_vec"), 0)
        if args.x_vec
        else None
    )
    adain = (
        np.mean(utils.get_pretrained_embedding(args.speaker, "adain"), 0)
        if args.adain
        else None
    )


    model = get_FastSpeech2(args.step).to(device)
    vocoder = utils.get_vocoder()

    speaker = speaker_to_id[args.speaker] if args.speaker_emb else None
    gst = np.mean(utils.get_gst(reference_mels, model), 0) if args.gst else None
    print(gst.shape, adain.shape)

    with torch.no_grad():
        for i in range(0, len(file_ids), hp.batch_size):
            synthesize(
                model,
                vocoder,
                d_vec,
                x_vec,
                adain,
                speaker,
                gst,
                texts[i : min(len(texts), i + hp.batch_size)],
                file_ids[i : min(len(file_ids), i + hp.batch_size)],
                None if mel_lens is None else mel_lens[i : min(len(file_ids), i + hp.batch_size)],
                None if Ds is None else Ds[i : min(len(file_ids), i + hp.batch_size)],
                None if f0s is None else f0s[i : min(len(file_ids), i + hp.batch_size)],
                None if energies is None else energies[i : min(len(file_ids), i + hp.batch_size)],
                prefix,
            )
