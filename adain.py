import torch
import yaml
import json
import os
import sys

from pathlib import Path
import numpy as np
from tqdm.auto import tqdm


def gen_adain(dataset_path, result_folder, adain_path='/root/AdaIN-VC'):
    file_prefix = 'VCTK-adain'

    sys.path = [os.getcwd()] + sys.path
    from model import AE
    from preprocess import get_spectrogram
    sys.path = sys.path[1:]

    assert torch.cuda.is_available()
    device = torch.device('cuda')

    with open(os.path.join(adain_path, '/config.yaml')) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    with open(os.path.join(adain_path, '/root/AdaIN-VC/vocoder/config.json')) as f:
        data_cfg = json.load(f)

    # load model
    model = AE(config).eval()
    model.load_state_dict(torch.load(os.path.join(adain_path, 'model.ckpt')))
    spk_encoder = model.speaker_encoder.to(device)

    # process datafolder
    os.makedirs(result_folder, exist_ok=True)
    for speaker_name in tqdm(os.listdir(dataset_path)):
        for root, _, filenames in os.walk(os.path.join(dataset_path, speaker_name)):
            for fn in filenames:
                if fn[-4:] != '.wav':
                    continue

                fn, mel = get_spectrogram(os.path.join(root, fn), data_cfg)
                with torch.no_grad():
                    emb = spk_encoder(torch.from_numpy(mel).T.unsqueeze(0).to(device)).squeeze(0).cpu()
                res_fn = f'{file_prefix}-{Path(fn).stem}.npy'
                # res_fn = f'{file_prefix}-{speaker_name}-{Path(fn).stem}.npy'
                np.save(os.path.join(result_folder, res_fn), emb)
