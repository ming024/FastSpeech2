import torch
import torch.nn as nn
import hparams as hp

class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, reduction='mean'):
        super(FastSpeech2Loss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)
        self.mae_loss = nn.L1Loss(reduction=reduction)

    def forward(self, d_predicted, d_target, p_predicted, p_target, e_predicted, e_target, mel, mel_postnet, mel_target):
        d_target.requires_grad = False
        p_target.requires_grad = False
        e_target.requires_grad = False
        mel_target.requires_grad = False

        mel_loss = self.mse_loss(mel, mel_target)
        mel_postnet_loss = self.mse_loss(mel_postnet, mel_target)

        d_loss = self.mae_loss(d_predicted, d_target.float())
        p_loss = self.mae_loss(p_predicted, p_target)
        e_loss = self.mae_loss(e_predicted, e_target)
        
        return mel_loss, mel_postnet_loss, d_loss, p_loss, e_loss
