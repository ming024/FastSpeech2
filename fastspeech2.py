import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.Models import Encoder, Decoder
from transformer.Layers import PostNet
from modules import VarianceAdaptor, Linear
import hparams as hp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self):
        super(FastSpeech2, self).__init__()

        self.encoder = Encoder()
        self.variance_adaptor = VarianceAdaptor()
        self.decoder = Decoder()

        self.mel_linear = Linear(hp.decoder_hidden, hp.n_mel_channels)
        self.postnet = PostNet()

    def forward(self, src_seq, src_pos, mel_pos=None, max_length=None, d_target=None, p_target=None, e_target=None):
        encoder_output, _ = self.encoder(src_seq, src_pos)
        
        if d_target is not None:
            variance_adaptor_output, d_prediction, p_prediction, e_prediction, _ = self.variance_adaptor(
                encoder_output, d_target, p_target, e_target, max_length)
        else:
            variance_adaptor_output, d_prediction, p_prediction, e_prediction, mel_pos = self.variance_adaptor(
                encoder_output, d_target, p_target, e_target, max_length)
        
        decoder_output = self.decoder(variance_adaptor_output, mel_pos)
        mel_output = self.mel_linear(decoder_output)
        mel_output_postnet = self.postnet(mel_output) + mel_output
        
        return mel_output, mel_output_postnet, d_prediction, p_prediction, e_prediction


if __name__ == "__main__":
    # Test
    model = FastSpeech2()
    print(model)
    print(sum(param.numel() for param in model.parameters()))
