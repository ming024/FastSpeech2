import os
import json
from string import ascii_lowercase

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FastSpeech2(nn.Module):
    """FastSpeech2"""

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.use_postnet = model_config["use_postnet"]

        if self.use_postnet:
            self.postnet = PostNet()

        self.speaker_emb = None
        self.language_emb = None

        if model_config["multilingual"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "languages.json"
                ),
                "r",
            ) as f:
                n_language = len(json.load(f))
            self.language_emb = nn.Embedding(
                n_language,
                model_config["transformer"]["encoder_hidden"],
            )

        if model_config["multi_speaker"]["use_multi_speaker"]:
            if (
                "embedding_type" in model_config["multi_speaker"]
                and model_config["multi_speaker"]["embedding_type"] == "vector"
                and model_config["multi_speaker"]["locations"]["encoder"]
            ):
                with open(
                    os.path.join(
                        preprocess_config["path"]["preprocessed_path"], "speakers.json"
                    ),
                    "r",
                ) as f:
                    speakers = json.load(f)
                self.raw_speaker_embs = {
                    v: np.load(
                        os.path.join(
                            preprocess_config["path"]["preprocessed_path"],
                            f"speaker-{k}.npy",
                        )
                    )
                    for k, v in speakers.items()
                }
                speaker_dim = len(list(self.raw_speaker_embs.values())[0][0])
                self.speaker_emb = nn.Linear(
                    speaker_dim,
                    model_config["transformer"]["encoder_hidden"],
                )
            else:
                with open(
                    os.path.join(
                        preprocess_config["path"]["preprocessed_path"], "speakers.json"
                    ),
                    "r",
                ) as f:
                    n_speaker = len(json.load(f))
                self.speaker_emb = nn.Embedding(
                    n_speaker,
                    model_config["transformer"]["encoder_hidden"],
                )

    def forward(
        self,
        languages,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        output = self.encoder(texts, src_masks)

        if self.language_emb is not None:
            output = output + self.language_emb(languages).unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        if self.speaker_emb is not None:
            if (
                "embedding_type" in self.model_config["multi_speaker"]
                and self.model_config["multi_speaker"]["use_multi_speaker"]
                and self.model_config["multi_speaker"]["embedding_type"] == "vector"
                and self.model_config["multi_speaker"]["locations"]["encoder"]
            ):
                speaker_tensor = np.array(
                    [self.raw_speaker_embs[x] for x in speakers.tolist()]
                )
                output = output + self.speaker_emb(
                    torch.FloatTensor(speaker_tensor).to(device)
                ).expand(-1, max_src_len, -1)
            else:
                output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                    -1, max_src_len, -1
                )

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        if self.use_postnet:
            postnet_output = self.postnet(output) + output
        else:
            postnet_output = output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )
