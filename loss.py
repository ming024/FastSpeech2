import torch
import torch.nn as nn
import hparams as hp

def mse_loss(prediction, target, length):
    batch_size = target.shape[0]
    loss = 0
    for p, t, l in zip(prediction, target, length):
        loss += torch.mean((prediction[:l]-target[:l])**2)
    loss /= batch_size
    return loss

def mae_loss(prediction, target, length):
    batch_size = target.shape[0]
    loss = 0
    for p, t, l in zip(prediction, target, length):
        loss += torch.mean(torch.abs(prediction[:l]-target[:l]))
    loss /= batch_size
    return loss

class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self):
        super(FastSpeech2Loss, self).__init__()

    def forward(self, d_predicted, d_target, p_predicted, p_target, e_predicted, e_target, mel, mel_postnet, mel_target, mel_length):
        d_target.requires_grad = False
        p_target.requires_grad = False
        e_target.requires_grad = False
        mel_target.requires_grad = False

        mel_loss = mse_loss(mel, mel_target, mel_length)
        mel_postnet_loss = mse_loss(mel_postnet, mel_target, mel_length)

        d_loss = mae_loss(d_predicted, d_target.float(), mel_length)
        p_loss = mae_loss(p_predicted, p_target, length)
        e_loss = mae_loss(e_predicted, e_target, length)
        
        return mel_loss, mel_postnet_loss, d_loss, p_loss, e_loss
