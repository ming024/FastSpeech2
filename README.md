# FastSpeech 2 for ICASSP 2021 M2VoC challenge

## Citing Us
```
@INPROCEEDINGS{chien2021investigating,
  author={Chien, Chung-Ming and Lin, Jheng-Hao and Huang, Chien-yu and Hsu, Po-chun and Lee, Hung-yi},
  booktitle={ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Investigating on Incorporating Pretrained and Learnable Speaker Representations for Multi-Speaker Multi-Style Text-to-Speech}, 
  year={2021},
  volume={},
  number={},
  pages={8588-8592},
  doi={10.1109/ICASSP39728.2021.9413880}
}
```

## Audio Samples
Audio samples which we submitted to ICASSP 2021 M2VoC challenge can be found [here](https://ming024.github.io/M2VoC/).  

## Dependencies
You can install the python dependencies with
```
pip3 install -r requirements.txt
```

## Data
Download the AIShell-3 and the M2VoC datasets, and set ``aishell3_path`` and ``m2voc_path`` in ``hparams.py`` to the paths to the datasets.

Run

```
python3 prepare_align.py
```

Then use [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/) (MFA) to align the wav files with the transcriptions.
The lexicon used in our work is put in ``text/pinyin-lexicon-r.txt``.
There is a problem with the pronunciation of the "ㄦ" character of the Chinese language in the lexicon, but not a big problem.
After aligning the utterances, put the resulted TextGrid files in ``hp.preprocessed_path/TextGrid``.

After that, run the preprocessing script by
```
python3 preprocess.py
```

After preprocessing, you will get a ``stat.txt`` file in your ``hp.preprocessed_path/``.
You have to modify the f0 and energy parameters in the ``hparams.py`` according to the content of ``stat.txt``.

We provide the pretrained speaker representations of the utterances in the AIShell-3 and the M2VoC datasets [here](https://drive.google.com/file/d/1VMH4LMHwnVR7c69fDKI8VJocHZK7AkHw/view?usp=sharing).
Extract the compressed files into ``hp.preprocessed_path``.

## Training

Train your model with
```
python3 train.py
```

You can use ``--x_vec``, ``--d_vec``, ``--adain``, ``--speaker_emb``, ``--gst`` to train your model with different pretrained or jointly-optimized speaker representations.
For example, if you wish to train a model combining d-vector and GST, try
```
python3 train.py --d_vec --gst
```


## Synthesis

Run
```
python3 generate.py --speaker SPEAKER_NAME --source SOURCE_PATH --step STEP ...
```

For example,
```
python3 generate.py --speaker  TST_T1_S5 --source preprocessed_data/M2VoC/Track1/TT_chat.txt --step 500000 --d_vec --x_vec --adain --speaker_emb --gst
```

The ``SOURCE`` files are available at ``preprocessed_data/M2VoC/Track*``

We provide several trained checkpoints [here](https://drive.google.com/drive/folders/1QduRkJB7shl61OkeuuuaWX2lJR6-khrI?usp=sharing). 
These models are trained with different combinations of speaker representations.
For example, the checkpoint named ``xvec_adain_checkpoint_500000.pth.tar`` requires X vector and AdaIN speaker representations, so you must specify ``--x_vec`` and ``--adain`` when synthesizing audio samples with this checkpoint.

MelGAN is used to convert the mel-spectrograms to the raw waveform in this repository.
We strongly recommend you to use WaveNet vocoder if audio quality is the first concern in your application.
