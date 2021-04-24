import os

import hparams as hp
from data import m2voc, aishell3


def main():
    # m2voc.prepare_align(hp.m2voc_path)
    aishell3.prepare_align(hp.aishell3_path)


if __name__ == "__main__":
    main()
