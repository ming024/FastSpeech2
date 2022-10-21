import argparse
import os

import torch
import yaml
import numpy as np

from utils.model import get_model
from utils.tools import to_device, get_mask_from_lengths
from synthesize import preprocess_english, preprocess_mandarin

from matplotlib import pyplot as plt
from text import _id_to_symbol

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_attention_map_dec(ids, attns_list, attn_map_path):
    """Get the figures of attention maps."""
    os.makedirs(attn_map_path, exist_ok=True)
    for layer_num in range(len(attns_list)):
        for head_num in range(attns_list[layer_num].shape[0]):
            attn_matrix = attns_list[layer_num][head_num].numpy()
            plt.figure(figsize=(10, 10))
            im = plt.imshow(attn_matrix, interpolation='none')
            im.axes.set_title('decoder, layer {}, head {}'.format(layer_num+1, head_num+1))
            # im.axes.set_xticks(range(len(phoneme_seq)))
            # im.axes.set_xticklabels(phoneme_seq, fontsize=200/len(phoneme_seq))
            # im.axes.set_yticks(range(len(phoneme_seq)))
            # im.axes.set_yticklabels(phoneme_seq, fontsize=200/len(phoneme_seq))
            im_cb = plt.colorbar(im)
            plt.savefig(os.path.join(attn_map_path, "{}_layer{}_head{}_dec.png".format(ids[0],layer_num+1, head_num+1)))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single"],   # only support single mode
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
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
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

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False)

    if args.mode == "single":
        ids = raw_texts = [args.text[:100]]
        speakers = np.array([args.speaker_id])
        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            texts = np.array([preprocess_english(args.text, preprocess_config)])
        elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
            texts = np.array([preprocess_mandarin(args.text, preprocess_config)])
        text_lens = np.array([len(texts[0])])
        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]

    batch = batchs[0]

    batch = to_device(batch, device)
    with torch.no_grad():
        # Forward
        ids = batch[0]
        texts = batch[3]
        speakers = batch[2]
        phoneme_seq = [_id_to_symbol[s].replace('@', '') for s in texts[0].numpy()]

        src_lens = batch[4]
        max_src_len = batch[5]
        src_masks = get_mask_from_lengths(src_lens, max_src_len)

        output, attns_list = model.encoder.forward(texts, src_masks, return_attns=True)

        if model.speaker_emb is not None:
            output = output +model.speaker_emb(speakers).unsqueeze(1).expand(
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
        ) = model.variance_adaptor(
            output,
            src_masks,
            p_control=args.pitch_control,
            e_control=args.energy_control,
            d_control=args.duration_control,
        )

        output, mel_masks, attns_list_dec = model.decoder(output, mel_masks, return_attns=True)

        attn_map_path = train_config['path']['log_path'].replace('log', 'attention_dec')
        print(attn_map_path)

        get_attention_map_dec(ids, attns_list_dec, attn_map_path)