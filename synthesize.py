import os
import re
import argparse
from string import punctuation
import json

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
from g2p import make_g2p
from pypinyin import pinyin, Style

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from dataset import TextDataset
from text import text_to_sequence
from text.symbols import MAPPINGS, TOKENIZERS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def preprocess_english(text, preprocess_config):

    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

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
            phones,
            preprocess_config["preprocessing"]["text"]["use_spe_features"],
            "eng",
        )
    )

    return np.array(sequence)


def preprocess_mandarin(text, preprocess_config):
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    phones = []
    pinyins = [
        p[0]
        for p in pinyin(
            text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
        )
    ]
    for p in pinyins:
        if p in lexicon:
            phones += lexicon[p]
        else:
            phones.append("sp")

    phones = "{" + " ".join(phones) + "}"
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones,
            preprocess_config["preprocessing"]["text"]["use_spe_features"],
            preprocess_config["preprocessing"]["text"]["language"],
        )
    )

    return np.array(sequence)


def preprocess(text, preprocess_config, lang):
    preprocessed_path = preprocess_config["path"]["preprocessed_path"]
    with open(os.path.join(preprocessed_path, "languages.json")) as f:
        language_map = json.load(f)
    language_map = {v: k for k, v in language_map.items()}
    lang = language_map[lang]
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])
    words = re.split(r"([,;.\-\?\!\s+])", text)
    g2p_norm = MAPPINGS[lang]["norm"]
    g2p = MAPPINGS[lang]["ipa"]
    phones = []
    for w in words:
        w = g2p_norm(w.lower()).output_string
        if w in lexicon:
            phones += lexicon[w]
        else:
            phones += TOKENIZERS[lang].tokenize(g2p(w).output_string)
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones,
            preprocess_config["preprocessing"]["text"]["use_spe_features"],
            lang,
        )
    )
    return np.array(sequence)


def synthesize(model, step, configs, vocoder, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values
    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control,
            )
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        required=True,
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--speaker_id",
        type=int,
        default=0,
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "--language_id",
        type=int,
        default=0,
        help="language ID for multi-lingual synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "-q", "--quick_config", type=str, required=False, help="config slug"
    )
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=False,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=False, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=False, help="path to train.yaml"
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    args = parser.parse_args()
    # Check source texts
    if args.mode == "batch":
        assert args.source is not None and args.text is None
    if args.mode == "single":
        assert args.source is None and args.text is not None

    # Read Config
    if args.quick_config:
        # Read Config
        preprocess_config = yaml.load(
            open(f"config/{args.quick_config}/preprocess.yaml", "r"),
            Loader=yaml.FullLoader,
        )
        model_config = yaml.load(
            open(f"config/{args.quick_config}/model.yaml", "r"), Loader=yaml.FullLoader
        )
        train_config = yaml.load(
            open(f"config/{args.quick_config}/train.yaml", "r"), Loader=yaml.FullLoader
        )
    else:
        # Read Config
        preprocess_config = yaml.load(
            open(args.preprocess_config, "r"), Loader=yaml.FullLoader
        )
        model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
        train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        dataset = TextDataset(args.source, preprocess_config)
        batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
        )
    if args.mode == "single":
        ids = raw_texts = [args.text[:100]]
        speakers = np.array([args.speaker_id])
        languages = np.array([args.language_id])
        preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        with open(os.path.join(preprocessed_path, "languages.json")) as f:
            language_map = json.load(f)
        language_map = {v: k for k, v in language_map.items()}
        lang = language_map[args.language_id]
        if lang == "eng":
            texts = np.array([preprocess_english(args.text, preprocess_config)])
        elif lang == "zh":
            texts = np.array([preprocess_mandarin(args.text, preprocess_config)])
        else:
            texts = np.array(
                [preprocess(args.text, preprocess_config, args.language_id)]
            )
        # TODO: replace this in config, confirmed this hparam is only used here, and so for languages other than en and zh, this is unnecessary
        # elif preprocess_config["preprocessing"]["text"]["language"] == "moh":
        #     texts = np.array([preprocess_mohawk(args.text, preprocess_config)])
        # elif preprocess_config["preprocessing"]["text"]["language"] == "git":
        #     texts = np.array([preprocess_gitksan(args.text, preprocess_config)])
        text_lens = np.array([len(texts[0])])
        batchs = [
            (ids, raw_texts, languages, speakers, texts, text_lens, max(text_lens))
        ]

    control_values = args.pitch_control, args.energy_control, args.duration_control

    synthesize(model, args.restore_step, configs, vocoder, batchs, control_values)
