# FastSpeech 2 - PyTorch Implementation

This repository is an extended PyTorch implementation of Microsoft's [**FastSpeech 2: Fast and High-Quality End-to-End Text to Speech**](https://arxiv.org/abs/2006.04558v1), initially based on [xcmyz's implementation](https://github.com/xcmyz/FastSpeech), with the core code structure derived from [ming024's original FastSpeech2 implementation](https://github.com/ming024/FastSpeech2).  
We introduce several modifications to enable training and inference using **phonological features** instead of phoneme IDs, supporting cross-lingual and low-resource speech synthesis scenarios. This modification allows more linguistically informed training and better generalization across languages. Using this version, we successfully trained a **German baseline TTS model**, and further performed **transfer learning** with a small amount of English data to train an English model.

Our method is inspired by the concept of using cross-lingual phonological information as described in the paper:  
> _"Cross-lingual Transfer of Phonological Features for Low-resource Speech Synthesis"_  
> [SSW11 Paper PDF](https://www.pure.ed.ac.uk/ws/portalfiles/portal/215873748/pf_tts_ssw11.pdf)

We also refer to the [PHOIBLE database](https://phoible.org) for phonological feature definitions and mappings.

The overall training and synthesis pipeline still follows the original repository structure [ming024's original FastSpeech2 implementation](https://github.com/ming024/FastSpeech2). However, we have made the following key modifications to support phonological feature-based modeling:

- **`text/` folder**: contains several modified files to support phonological feature data preparation.
- **`transformer/models.py`**: updated to allow model input as phonological feature vectors instead of phoneme IDs.
- **`synthesis.py`**: modified to support inference using phonological features as input.

---

# References
- [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558), Y. Ren, *et al*.
- [xcmyz's FastSpeech implementation](https://github.com/xcmyz/FastSpeech)
- [TensorSpeech's FastSpeech 2 implementation](https://github.com/TensorSpeech/TensorflowTTS)
- [rishikksh20's FastSpeech 2 implementation](https://github.com/rishikksh20/FastSpeech2)
- [PHOIBLE: Phonological Segment Inventory Database](https://phoible.org)
- [Cross-lingual Transfer of Phonological Features for Low-resource Speech Synthesis (SSW11)](https://www.pure.ed.ac.uk/ws/portalfiles/portal/215873748/pf_tts_ssw11.pdf)
