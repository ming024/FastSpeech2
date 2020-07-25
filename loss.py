import torch
import torch.nn as nn
import hparams as hp

class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self):
        super(FastSpeech2Loss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, log_d_predicted, log_d_target, p_predicted, p_target, e_predicted, e_target, mel, mel_postnet, mel_target, src_mask, mel_mask):
        log_d_target.requires_grad = False
        p_target.requires_grad = False
        e_target.requires_grad = False
        mel_target.requires_grad = False
        
        log_d_predicted = log_d_predicted.masked_select(src_mask)
        log_d_target = log_d_target.masked_select(src_mask)
        p_predicted = p_predicted.masked_select(mel_mask)
        p_target = p_target.masked_select(mel_mask)
        e_predicted = e_predicted.masked_select(mel_mask)
        e_target = e_target.masked_select(mel_mask)

        mel = mel.masked_select(mel_mask.unsqueeze(-1))
        mel_postnet = mel_postnet.masked_select(mel_mask.unsqueeze(-1))
        mel_target = mel_target.masked_select(mel_mask.unsqueeze(-1))

        mel_loss = self.mse_loss(mel, mel_target)
        mel_postnet_loss = self.mse_loss(mel_postnet, mel_target)

        d_loss = self.mae_loss(log_d_predicted, log_d_target)
        p_loss = self.mae_loss(p_predicted, p_target)
        e_loss = self.mae_loss(e_predicted, e_target)
        
        return mel_loss, mel_postnet_loss, d_loss, p_loss, e_loss
