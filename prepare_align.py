import os
from data import ljspeech, blizzard2013
import hparams as hp

def main():
    in_dir = hp.data_path

    if hp.dataset == "LJSpeech":
        ljspeech.prepare_align(in_dir)
    if hp.dataset == "Blizzard2013":
        blizzard2013.prepare_align(in_dir)
    
if __name__ == "__main__":
    main()
