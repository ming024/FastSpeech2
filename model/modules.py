import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch import bucketize as f0_bucketize
from torch import bucketize as energy_bucketize

import utils
import hparams as hp
from transformer.SubLayers import MultiHeadAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def masked_mean(tensor, mask):
    return (
        (torch.sum(tensor * mask, 1) / torch.sum(mask, 1))
        .unsqueeze(1)
        .expand(-1, mask.shape[1])
    )


def masked_std(tensor, mask):
    squared_error = (tensor - masked_mean(tensor, mask)) ** 2
    return torch.sqrt(masked_mean(squared_error, mask))


class VarianceAdaptor(nn.Module):
    """ Variance Adaptor """

    def __init__(self):
        super().__init__()
        self.duration_predictor = VariancePredictor()
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor()
        self.energy_predictor = VariancePredictor()

        self.pitch_bins = nn.Parameter(
            torch.linspace(hp.f0_min, hp.f0_max, hp.n_bins - 1),
            requires_grad=False,
        )
        self.energy_bins = nn.Parameter(
            torch.linspace(hp.energy_min, hp.energy_max, hp.n_bins - 1),
            requires_grad=False,
        )

        self.pitch_embedding = nn.Embedding(hp.n_bins, hp.encoder_hidden)
        self.energy_embedding = nn.Embedding(hp.n_bins, hp.encoder_hidden)

    def forward(
        self,
        x,
        src_mask,
        mel_mask=None,
        duration_target=None,
        pitch_target=None,
        energy_target=None,
        max_len=None,
    ):

        log_duration_prediction = self.duration_predictor(x, src_mask)
        pitch_prediction = self.pitch_predictor(x, src_mask)
        if pitch_target is not None:
            pitch_embedding = self.pitch_embedding(
                f0_bucketize(pitch_target, self.pitch_bins)
            )
        else:
            pitch_prediction = pitch_prediction
            pitch_embedding = self.pitch_embedding(
                f0_bucketize(pitch_prediction, self.pitch_bins)
            )

        energy_prediction = self.energy_predictor(x, src_mask)
        if energy_target is not None:
            energy_embedding = self.energy_embedding(
                energy_bucketize(energy_target, self.energy_bins)
            )
        else:
            energy_prediction = energy_prediction
            energy_embedding = self.energy_embedding(
                energy_bucketize(energy_prediction, self.energy_bins)
            )

        x = x + pitch_embedding + energy_embedding

        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            duration_rounded = torch.clamp(
                torch.round(torch.exp(log_duration_prediction) - hp.log_offset),
                min=0,
            )
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = utils.get_mask_from_lengths(mel_len)

        return (
            x,
            log_duration_prediction,
            duration_rounded,
            pitch_prediction,
            energy_prediction,
            mel_len,
            mel_mask,
        )


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self):
        super().__init__()

    # def LR(self, x, duration, max_len):
    #     batch_size = x.shape[0]
    #     channel_size = x.shape[2]
    #     mel_len = torch.sum(duration, 1)
    #     expanded = torch.zeros(
    #         (batch_size, torch.max(mel_len), channel_size), dtype=torch.float
    #     ).to(device)
    #     for i in range(batch_size):
    #         self.expand(x[i], duration[i], expanded[i])

    #     return expanded, mel_len

    # def expand(self, source, lengths, target):
    #     pos = 0
    #     for i, l in enumerate(lengths):
    #         target[pos : pos + l] = source[i]
    #         pos += l

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = utils.pad(output, max_len)
        else:
            output = utils.pad(output)

        return output, torch.LongTensor(mel_len).to(device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(int(expand_size), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class VariancePredictor(nn.Module):
    """ Duration, Pitch and Energy Predictor """

    def __init__(self):
        super().__init__()

        self.input_size = hp.encoder_hidden
        self.filter_size = hp.variance_predictor_filter_size
        self.kernel = hp.variance_predictor_kernel_size
        self.conv_output_size = hp.variance_predictor_filter_size
        self.dropout = hp.variance_predictor_dropout

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x


class ReferenceEncoder(nn.Module):
    """
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    """

    def __init__(self):
        super().__init__()
        self.filter_size = [1] + hp.ref_filters
        self.dropout = hp.encoder_dropout
        self.conv = nn.Sequential(
            OrderedDict(
                [
                    module
                    for i in range(len(hp.ref_filters))
                    for module in (
                        (
                            "conv2d_{}".format(i + 1),
                            Conv2d(
                                in_channels=self.filter_size[i],
                                out_channels=self.filter_size[i + 1],
                                kernel_size=(3, 3),
                                stride=(2, 2),
                                padding=(1, 1),
                            ),
                        ),
                        ("relu_{}".format(i + 1), nn.ReLU()),
                        (
                            "layer_norm_{}".format(i + 1),
                            nn.LayerNorm(self.filter_size[i + 1]),
                        ),
                        ("dropout_{}".format(i + 1), nn.Dropout(self.dropout)),
                    )
                ]
            )
        )

        self.gru = nn.GRU(
            input_size=hp.ref_filters[-1] * 2,
            hidden_size=hp.ref_gru_hidden,
            batch_first=True,
        )

    def forward(self, inputs):
        out = inputs.unsqueeze(3)
        out = self.conv(out)
        out = out.view(out.shape[0], out.shape[1], -1).contiguous()
        self.gru.flatten_parameters()
        memory, out = self.gru(out)

        return out.squeeze(0)


class StyleAttention(nn.Module):
    def __init__(self):

        super().__init__()
        self.input_size = hp.ref_gru_hidden
        self.output_size = hp.gst_size
        self.n_token = hp.n_style_token
        self.n_head = hp.n_style_attn_head
        self.token_size = self.output_size // self.n_head

        self.tokens = nn.Parameter(torch.FloatTensor(self.n_token, self.token_size))

        self.q_linear = nn.Linear(self.input_size, self.output_size)
        self.k_linear = nn.Linear(self.token_size, self.output_size)
        self.v_linear = nn.Linear(self.token_size, self.output_size)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)
        self.temperature = (self.output_size // self.n_head) ** 0.5
        nn.init.normal_(self.tokens)

    def forward(self, inputs, token_id=None):
        bs = inputs.size(0)
        q = self.q_linear(inputs.unsqueeze(1))
        k = self.k_linear(self.tanh(self.tokens).unsqueeze(0).expand(bs, -1, -1))
        v = self.v_linear(self.tanh(self.tokens).unsqueeze(0).expand(bs, -1, -1))

        q = q.view(bs, q.shape[1], self.n_head, self.token_size)
        k = k.view(bs, k.shape[1], self.n_head, self.token_size)
        v = v.view(bs, v.shape[1], self.n_head, self.token_size)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, q.shape[1], q.shape[3])
        k = k.permute(2, 0, 3, 1).contiguous().view(-1, k.shape[3], k.shape[1])
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, v.shape[1], v.shape[3])

        scores = torch.bmm(q, k) / self.temperature
        scores = self.softmax(scores)
        if token_id is not None:
            scores = torch.zeros_like(scores)
            scores[:, :, token_id] = 1

        style_emb = torch.bmm(scores, v).squeeze(1)
        style_emb = style_emb.contiguous().view(self.n_head, bs, self.token_size)
        style_emb = style_emb.permute(1, 0, 2).contiguous().view(bs, -1)

        return style_emb


class Conv2d(nn.Module):
    """
    Convolution 2D Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 3)
        x = x.contiguous().transpose(2, 3)
        x = self.conv(x)
        x = x.contiguous().transpose(2, 3)
        x = x.contiguous().transpose(1, 3)

        return x
