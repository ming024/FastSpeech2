import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import argparse
import os
import time

from fastspeech2 import FastSpeech2
from loss import FastSpeech2Loss
from dataset import Dataset
from optimizer import ScheduledOptim
from evaluate import evaluate
import hparams as hp
import utils
import audio as Audio
import waveglow

def main(args):
    torch.manual_seed(0)

    # Get device
    device = torch.device('cuda'if torch.cuda.is_available()else 'cpu')
    
    # Get dataset
    dataset = Dataset("train.txt") 
    loader = DataLoader(dataset, batch_size=hp.batch_size**2, shuffle=True, 
        collate_fn=dataset.collate_fn, drop_last=True, num_workers=0)

    # Define model
    model = nn.DataParallel(FastSpeech2()).to(device)
    print("Model Has Been Defined")
    num_param = utils.get_param_num(model)
    print('Number of FastSpeech2 Parameters:', num_param)

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), betas=hp.betas, eps=hp.eps, weight_decay = hp.weight_decay)
    scheduled_optim = ScheduledOptim(optimizer, hp.decoder_hidden, hp.n_warm_up_step, args.restore_step)
    Loss = FastSpeech2Loss().to(device) 
    print("Optimizer and Loss Function Defined.")

    # Load checkpoint if exists
    checkpoint_path = os.path.join(hp.checkpoint_path)
    try:
        checkpoint = torch.load(os.path.join(
            checkpoint_path, 'checkpoint_{}.pth.tar'.format(args.restore_step)))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("\n---Model Restored at Step {}---\n".format(args.restore_step))
    except:
        print("\n---Start New Training---\n")
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

    # Load vocoder
    wave_glow = utils.get_WaveGlow()

    # Init logger
    log_path = hp.log_path
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logger = SummaryWriter(log_path)

    # Init synthesis directory
    synth_path = hp.synth_path
    if not os.path.exists(synth_path):
        os.makedirs(synth_path)

    # Define Some Information
    Time = np.array([])
    Start = time.perf_counter()
    
    # Training
    model = model.train()
    for epoch in range(hp.epochs):
        # Get Training Loader
        total_step = hp.epochs * len(loader) * hp.batch_size

        for i, batchs in enumerate(loader):
            for j, data_of_batch in enumerate(batchs):
                start_time = time.perf_counter()

                current_step = i*hp.batch_size + j + args.restore_step + epoch*len(loader)*hp.batch_size + 1

                # Init
                scheduled_optim.zero_grad()

                # Get Data
                text = torch.from_numpy(data_of_batch["text"]).long().to(device)
                mel_target = torch.from_numpy(data_of_batch["mel_target"]).float().to(device)
                D = torch.from_numpy(data_of_batch["D"]).int().to(device)
                f0 = torch.from_numpy(data_of_batch["f0"]).float().to(device)
                energy = torch.from_numpy(data_of_batch["energy"]).float().to(device)
                mel_pos = torch.from_numpy(data_of_batch["mel_pos"]).long().to(device)
                src_pos = torch.from_numpy(data_of_batch["src_pos"]).long().to(device)
                src_len = torch.from_numpy(data_of_batch["src_len"]).long().to(device)
                mel_len = torch.from_numpy(data_of_batch["mel_len"]).long().to(device)
                max_len = max(data_of_batch["mel_len"]).astype(np.int16)
                
                # Forward
                mel_output, mel_postnet_output, duration_output, f0_output, energy_output = model(
                    text, src_pos, mel_pos, max_len, D, f0, energy)
                
                # Cal Loss
                mel_loss, mel_postnet_loss, d_loss, f_loss, e_loss = Loss(
                        duration_output, D, f0_output, f0, energy_output, energy, mel_output, mel_postnet_output, mel_target, src_len, mel_len)
                total_loss = mel_loss + mel_postnet_loss + d_loss + f_loss + e_loss
                 
                # Logger
                t_l = total_loss.item()
                m_l = mel_loss.item()
                m_p_l = mel_postnet_loss.item()
                d_l = d_loss.item()
                f_l = f_loss.item()
                e_l = e_loss.item()
                with open(os.path.join(log_path, "total_loss.txt"), "a") as f_total_loss:
                    f_total_loss.write(str(t_l)+"\n")
                with open(os.path.join(log_path, "mel_loss.txt"), "a") as f_mel_loss:
                    f_mel_loss.write(str(m_l)+"\n")
                with open(os.path.join(log_path, "mel_postnet_loss.txt"), "a") as f_mel_postnet_loss:
                    f_mel_postnet_loss.write(str(m_p_l)+"\n")
                with open(os.path.join(log_path, "duration_loss.txt"), "a") as f_d_loss:
                    f_d_loss.write(str(d_l)+"\n")
                with open(os.path.join(log_path, "f0_loss.txt"), "a") as f_f_loss:
                    f_f_loss.write(str(f_l)+"\n")
                with open(os.path.join(log_path, "energy_loss.txt"), "a") as f_e_loss:
                    f_e_loss.write(str(e_l)+"\n")
                 
                # Backward
                total_loss.backward()

                # Clipping gradients to avoid gradient explosion
                nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip_thresh)

                # Update weights
                scheduled_optim.step_and_update_lr()
                
                # Print
                if current_step % hp.log_step == 0:
                    Now = time.perf_counter()

                    str1 = "Epoch [{}/{}], Step [{}/{}]:".format(
                        epoch+1, hp.epochs, current_step, total_step)
                    str2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Duration Loss: {:.4f}, F0 Loss: {:.4f}, Energy Loss: {:.4f};".format(
                        t_l, m_l, m_p_l, d_l, f_l, e_l)
                    str3 = "Time Used: {:.3f}s, Estimated Time Remaining: {:.3f}s.".format(
                        (Now-Start), (total_step-current_step)*np.mean(Time))

                    print("\n" + str1)
                    print(str2)
                    print(str3)
                    
                    with open(os.path.join(log_path, "log.txt"), "a") as f_log:
                        f_log.write(str1 + "\n")
                        f_log.write(str2 + "\n")
                        f_log.write(str3 + "\n")
                        f_log.write("\n")

                    logger.add_scalars('Loss/total_loss', {'training': t_l}, current_step)
                    logger.add_scalars('Loss/mel_loss', {'training': m_l}, current_step)
                    logger.add_scalars('Loss/mel_postnet_loss', {'training': m_p_l}, current_step)
                    logger.add_scalars('Loss/duration_loss', {'training': d_l}, current_step)
                    logger.add_scalars('Loss/F0_loss', {'training': f_l}, current_step)
                    logger.add_scalars('Loss/energy_loss', {'training': e_l}, current_step)
                
                if current_step % hp.save_step == 0:
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                    )}, os.path.join(checkpoint_path, 'checkpoint_{}.pth.tar'.format(current_step)))
                    print("save model at step {} ...".format(current_step))

                if current_step % hp.synth_step == 0:
                    length = mel_len[0].item()
                    mel_target_torch = mel_target[0, :length].detach().unsqueeze(0).transpose(1, 2)
                    mel_target = mel_target[0, :length].detach().cpu().transpose(0, 1)
                    mel_torch = mel_output[0, :length].detach().unsqueeze(0).transpose(1, 2)
                    mel = mel_output[0, :length].detach().cpu().transpose(0, 1)
                    mel_postnet_torch = mel_postnet_output[0, :length].detach().unsqueeze(0).transpose(1, 2)
                    mel_postnet = mel_postnet_output[0, :length].detach().cpu().transpose(0, 1)
                    Audio.tools.inv_mel_spec(mel, os.path.join(synth_path, "step_{}_griffin_lim.wav".format(current_step)))
                    Audio.tools.inv_mel_spec(mel_postnet, os.path.join(synth_path, "step_{}_postnet_griffin_lim.wav".format(current_step)))
                    waveglow.inference.inference(mel_torch, wave_glow, os.path.join(synth_path, "step_{}_waveglow.wav".format(current_step)))
                    waveglow.inference.inference(mel_postnet_torch, wave_glow, os.path.join(synth_path, "step_{}_postnet_waveglow.wav".format(current_step)))
                    waveglow.inference.inference(mel_target_torch, wave_glow, os.path.join(synth_path, "step_{}_ground-truth_waveglow.wav".format(current_step)))
                    
                    f0 = f0[0, :length].detach().cpu().numpy()
                    energy = energy[0, :length].detach().cpu().numpy()
                    f0_output = f0_output[0, :length].detach().cpu().numpy()
                    energy_output = energy_output[0, :length].detach().cpu().numpy()
                    
                    utils.plot_data([(mel_postnet.numpy(), f0_output, energy_output), (mel_target.numpy(), f0, energy)], 
                        ['Synthetized Spectrogram', 'Ground-Truth Spectrogram'], filename=os.path.join(synth_path, 'step_{}.png'.format(current_step)))
                
                if current_step % hp.eval_step == 0:
                    model.eval()
                    with torch.no_grad():
                        d_l, f_l, e_l, m_l, m_p_l = evaluate(model, current_step)
                        t_l = d_l + f_l + e_l + m_l + m_p_l
                        
                        logger.add_scalars('Loss/total_loss', {'validation': t_l}, current_step)
                        logger.add_scalars('Loss/mel_loss', {'validation': m_l}, current_step)
                        logger.add_scalars('Loss/mel_postnet_loss', {'validation': m_p_l}, current_step)
                        logger.add_scalars('Loss/duration_loss', {'validation': d_l}, current_step)
                        logger.add_scalars('Loss/F0_loss', {'validation': f_l}, current_step)
                        logger.add_scalars('Loss/energy_loss', {'validation': e_l}, current_step)

                    model.train()

                end_time = time.perf_counter()
                Time = np.append(Time, end_time - start_time)
                if len(Time) == hp.clear_Time:
                    temp_value = np.mean(Time)
                    Time = np.delete(
                        Time, [i for i in range(len(Time))], axis=None)
                    Time = np.append(Time, temp_value)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type=int, default=0)
    args = parser.parse_args()

    main(args)
