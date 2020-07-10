import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import re
from g2p_en import G2p

from fastspeech2 import FastSpeech2
from text import text_to_sequence, sequence_to_text
import hparams as hp
import utils
import audio as Audio
import waveglow

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess(text):
    g2p = G2p()
    phone = g2p(text)
    phone = list(filter(lambda p: p != ' ', phone))
    phone = '{'+ '}{'.join(phone) + '}' # '{A}{B}{$}{C}', $ represents silent phones
    phone = re.sub(r'\{[^\w\s]?\}', ' ', phone)
    phone = phone.replace('}{', ' ')
    
    print('|' + phone + '|')    
    sequence = np.array(text_to_sequence(phone, hp.text_cleaners))
    sequence = np.stack([sequence])

    return torch.from_numpy(sequence).long().to(device)

def get_FastSpeech2(num):
    checkpoint_path = os.path.join(hp.checkpoint_path, "checkpoint_{}.pth.tar".format(num))
    model = nn.DataParallel(FastSpeech2())
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    model.requires_grad = False
    model.eval()
    return model

def synthesize(model, text, sentence, prefix=''):
    src_pos = np.array([i+1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    src_pos = torch.from_numpy(src_pos).to(device).long()
        
    model.to(device)
    mel, mel_postnet, duration_output, f0_output, energy_output = model(text, src_pos)
    model.to('cpu')
    
    mel_torch = mel.transpose(1, 2).detach()
    mel_postnet_torch = mel_postnet.transpose(1, 2).detach()
    mel = mel[0].cpu().transpose(0, 1).detach()
    mel_postnet = mel_postnet[0].cpu().transpose(0, 1).detach()
    f0_output = f0_output[0].detach().cpu().numpy()
    energy_output = energy_output[0].detach().cpu().numpy()

    if not os.path.exists(hp.test_path):
        os.makedirs(hp.test_path)

    Audio.tools.inv_mel_spec(mel_postnet, os.path.join(hp.test_path, '{}_griffin_lim_{}.wav'.format(prefix, sentence)))
    wave_glow = utils.get_WaveGlow()
    waveglow.inference.inference(mel_postnet_torch, wave_glow, os.path.join(
        hp.test_path, '{}_waveglow_{}.wav'.format(prefix, sentence)))

    utils.plot_data([(mel_postnet.numpy(), f0_output, energy_output)], ['Synthesized Spectrogram'], filename=os.path.join(hp.test_path, '{}_{}.png'.format(prefix, sentence)))


if __name__ == "__main__":
    # Test
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=30000)
    args = parser.parse_args()
    
    sentence = "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition"
    sentence = "in being comparatively modern."
    sentence = "For although the Chinese took impressions from wood blocks engraved in relief for centuries before the woodcutters of the Netherlands, by a similar process"
    sentence = "produced the block books, which were the immediate predecessors of the true printed book,"
    sentence = "the invention of movable metal letters in the middle of the fifteenth century may justly be considered as the invention of the art of printing."
    sentence = "And it is worth mention in passing that, as an example of fine typography,"
    sentence = "the earliest book printed with movable types, the Gutenberg, or \"forty-two line Bible\" of about 1455,"
    sentence = "has never been surpassed."
    sentence = "Printing, then, for our purpose, may be considered as the art of making books by means of movable types."
    sentence = "Now, as all books not primarily intended as picture-books consist principally of types composed to form letterpress,"

    text = preprocess(sentence)
    model = get_FastSpeech2(args.step)
    synthesize(model, text, sentence, prefix='step_{}'.format(args.step))
