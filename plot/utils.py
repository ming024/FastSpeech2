import matplotlib
import numpy as np
from matplotlib import pyplot as plt

import hparams as hp


def plot_mel(data, titles, filename):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]

    def add_axis(fig, old_ax, offset=0):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    def expand(values, durations):
        out = list()
        for value, d in zip(values, durations):
            out += [value] * d
        return np.array(out)

    for i in range(len(data)):
        spectrogram, pitch, energy, duration = data[i]
        pitch = expand(pitch, duration) * hp.f0_std + hp.f0_mean
        energy = expand(energy, duration)
        axes[i][0].imshow(spectrogram, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, hp.n_mel_channels)
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

        ax1 = add_axis(fig, axes[i][0])
        ax1.plot(pitch, color="tomato")
        ax1.set_xlim(0, spectrogram.shape[1])
        ax1.set_ylim(
            hp.f0_min * hp.f0_std + hp.f0_mean, hp.f0_max * hp.f0_std + hp.f0_mean
        )
        ax1.set_ylabel("F0", color="tomato")
        ax1.tick_params(
            labelsize="x-small", colors="tomato", bottom=False, labelbottom=False
        )

        ax2 = add_axis(fig, axes[i][0], 1.2)
        ax2.plot(energy, color="darkviolet")
        ax2.set_xlim(0, spectrogram.shape[1])
        ax2.set_ylim(hp.energy_min, hp.energy_max)
        ax2.set_ylabel("Energy", color="darkviolet")
        ax2.yaxis.set_label_position("right")
        ax2.tick_params(
            labelsize="x-small",
            colors="darkviolet",
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False,
            right=True,
            labelright=True,
        )

    plt.savefig(filename, dpi=200)
    plt.close()