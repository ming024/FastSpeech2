from collections import OrderedDict

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import hparams as hp
from transformer.Models import Encoder, Decoder
from transformer.Layers import PostNet
from utils import get_mask_from_lengths
from .modules import VarianceAdaptor, ReferenceEncoder, StyleAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, speaker_num=1):
        super(FastSpeech2, self).__init__()

        self.encoder = Encoder()

        # Pretrained speaker representations
        self.d_vec_proj = nn.Sequential(
            OrderedDict(
                [
                    (
                        "linear_1",
                        nn.Linear(hp.d_vec_size, hp.encoder_hidden),
                    ),
                    ("relu_1", nn.ReLU()),
                    (
                        "linear_2",
                        nn.Linear(hp.encoder_hidden, hp.encoder_hidden),
                    ),
                ]
            )
        )
        self.x_vec_proj = nn.Sequential(
            OrderedDict(
                [
                    (
                        "linear_1",
                        nn.Linear(hp.x_vec_size, hp.encoder_hidden),
                    ),
                    ("relu_1", nn.ReLU()),
                    (
                        "linear_2",
                        nn.Linear(hp.encoder_hidden, hp.encoder_hidden),
                    ),
                ]
            )
        )
        self.adain_proj = nn.Sequential(
            OrderedDict(
                [
                    (
                        "linear_1",
                        nn.Linear(hp.adain_emb_size, hp.encoder_hidden),
                    ),
                    ("relu_1", nn.ReLU()),
                    (
                        "linear_2",
                        nn.Linear(hp.encoder_hidden, hp.encoder_hidden),
                    ),
                ]
            )
        )

        # Jointly trained speaker representations
        self.speaker_embedding = nn.Embedding(speaker_num, hp.speaker_emb_size)
        self.speaker_proj = nn.Sequential(
            OrderedDict(
                [
                    (
                        "linear_1",
                        nn.Linear(hp.speaker_emb_size, hp.encoder_hidden),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("linear_2", nn.Linear(hp.encoder_hidden, hp.encoder_hidden)),
                ]
            )
        )

        # GST
        self.reference_encoder = ReferenceEncoder()
        self.style_attention = StyleAttention()
        self.gst_proj = nn.Sequential(
            OrderedDict(
                [
                    (
                        "linear_1",
                        nn.Linear(hp.gst_size, hp.encoder_hidden),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("linear_2", nn.Linear(hp.encoder_hidden, hp.encoder_hidden)),
                ]
            )
        )

        self.variance_adaptor = VarianceAdaptor()

        self.decoder = Decoder()
        self.mel_linear = nn.Linear(hp.decoder_hidden, hp.n_mel_channels)

        self.postnet = PostNet()

    def forward(
        self,
        src_seq,
        src_len,
        mel_len=None,
        d_target=None,
        p_target=None,
        e_target=None,
        mel_target=None,
        max_src_len=None,
        max_mel_len=None,
        d_vec=None,
        x_vec=None,
        adain=None,
        speaker=None,
        use_gst=False,
        gst=None,
    ):
        start_time = time.perf_counter()
        src_mask = get_mask_from_lengths(src_len, max_src_len)
        mel_mask = (
            get_mask_from_lengths(mel_len, max_mel_len) if mel_len is not None else None
        )

        encoder_output = self.encoder(src_seq, src_mask)

        if d_vec is not None:
            encoder_output = encoder_output + self.d_vec_proj(d_vec).unsqueeze(
                1
            ).expand(-1, max_src_len, -1)
        if x_vec is not None:
            encoder_output = encoder_output + self.x_vec_proj(x_vec).unsqueeze(
                1
            ).expand(-1, max_src_len, -1)
        if adain is not None:
            encoder_output = encoder_output + self.adain_proj(adain).unsqueeze(
                1
            ).expand(-1, max_src_len, -1)
        if speaker is not None:
            encoder_output = encoder_output + self.speaker_proj(
                self.speaker_embedding(speaker)
            ).unsqueeze(1).expand(-1, max_src_len, -1)

        if use_gst:
            if gst is None:
                style_embedding = self.reference_encoder(mel_target)
                gst = self.style_attention(style_embedding)

            encoder_output = encoder_output + self.gst_proj(gst).unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        if d_target is not None:
            (
                variance_adaptor_output,
                d_prediction,
                d_rounded,
                p_prediction,
                e_prediction,
                _,
                _,
            ) = self.variance_adaptor(
                encoder_output,
                src_mask,
                mel_mask,
                d_target,
                p_target,
                e_target,
                max_mel_len,
            )
        else:
            (
                variance_adaptor_output,
                d_prediction,
                d_rounded,
                p_prediction,
                e_prediction,
                mel_len,
                mel_mask,
            ) = self.variance_adaptor(
                encoder_output,
                src_mask,
                mel_mask,
                d_target,
                p_target,
                e_target,
                max_mel_len,
            )

        decoder_output = self.decoder(variance_adaptor_output, mel_mask)
        mel_output = self.mel_linear(decoder_output)

        mel_output_postnet = self.postnet(mel_output) + mel_output

        return (
            mel_output,
            mel_output_postnet,
            d_prediction,
            d_rounded,
            p_prediction,
            e_prediction,
            src_mask,
            mel_mask,
            mel_len,
        )