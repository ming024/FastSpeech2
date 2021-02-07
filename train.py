import argparse
import os
import time

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import utils
import hparams as hp
import audio as Audio
from dataset import Dataset
from model.fastspeech2 import FastSpeech2
from model.loss import FastSpeech2Loss
from model.optimizer import ScheduledOptim
from plot.utils import plot_mel


def main(args):
    torch.manual_seed(0)

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get dataset
    dataset = Dataset("train.txt")
    loader = DataLoader(
        dataset,
        batch_size=hp.batch_size ** 2,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        drop_last=True,
        num_workers=0,
    )

    # Define model
    speaker_num = len(utils.get_speaker_to_id())
    model = nn.DataParallel(FastSpeech2(speaker_num)).to(device)
    print("Model Has Been Defined")
    num_param = utils.get_param_num(model)
    print("Number of FastSpeech2 Parameters:", num_param)

    # Optimizer and loss
    optimizer = torch.optim.Adam(
        model.parameters(), betas=hp.betas, eps=hp.eps, weight_decay=hp.weight_decay
    )
    scheduled_optim = ScheduledOptim(
        optimizer,
        hp.decoder_hidden,
        hp.n_warm_up_step,
        hp.aneal_steps,
        hp.aneal_rate,
        args.restore_step,
    )
    Loss = FastSpeech2Loss().to(device)
    print("Optimizer and Loss Function Defined.")

    # Load checkpoint if exists
    checkpoint_path = hp.checkpoint_path
    try:
        checkpoint = torch.load(
            os.path.join(
                checkpoint_path, "checkpoint_{}.pth.tar".format(args.restore_step)
            )
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("\n---Model Restored at Step {}---\n".format(args.restore_step))
    except:
        print("\n---Start New Training---\n")
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

    # Load vocoder
    vocoder = utils.get_vocoder()

    # Init logger
    log_path = hp.log_path
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        os.makedirs(os.path.join(log_path, "train"))
    train_logger = SummaryWriter(os.path.join(log_path, "train"))

    # Init synthesis directory
    synth_path = hp.synth_path
    if not os.path.exists(synth_path):
        os.makedirs(synth_path)

    # Define Some Information
    Time = np.array([])
    Start = time.perf_counter()

    # Training
    model = model.train()
    total_step = hp.epochs * len(loader) * hp.batch_size
    for epoch in range(hp.epochs):
        for i, batchs in enumerate(loader):
            for j, data_of_batch in enumerate(batchs):
                start_time = time.perf_counter()

                current_step = (
                    i * hp.batch_size
                    + j
                    + args.restore_step
                    + epoch * len(loader) * hp.batch_size
                    + 1
                )

                # Get Data
                id_ = data_of_batch["id"]
                speaker = torch.from_numpy(data_of_batch["speaker"]).long().to(device)
                text = torch.from_numpy(data_of_batch["text"]).long().to(device)
                mel_target = (
                    torch.from_numpy(data_of_batch["mel_target"]).float().to(device)
                )
                D = torch.from_numpy(data_of_batch["D"]).long().to(device)
                log_D = torch.from_numpy(data_of_batch["log_D"]).float().to(device)
                f0 = torch.from_numpy(data_of_batch["f0"]).float().to(device)
                energy = torch.from_numpy(data_of_batch["energy"]).float().to(device)
                src_len = torch.from_numpy(data_of_batch["src_len"]).long().to(device)
                mel_len = torch.from_numpy(data_of_batch["mel_len"]).long().to(device)
                max_src_len = np.max(data_of_batch["src_len"]).astype(np.int32)
                max_mel_len = np.max(data_of_batch["mel_len"]).astype(np.int32)
                d_vec = torch.from_numpy(data_of_batch["d_vec"]).float().to(device)
                x_vec = torch.from_numpy(data_of_batch["x_vec"]).float().to(device)
                adain = torch.from_numpy(data_of_batch["adain"]).float().to(device)

                # Forward
                (
                    mel_output,
                    mel_postnet_output,
                    log_duration_output,
                    _,
                    f0_output,
                    energy_output,
                    src_mask,
                    mel_mask,
                    _,
                ) = model(
                    text,
                    src_len,
                    mel_len,
                    D,
                    f0,
                    energy,
                    mel_target,
                    max_src_len,
                    max_mel_len,
                    d_vec=d_vec if args.d_vec else None,
                    x_vec=x_vec if args.x_vec else None,
                    adain=adain if args.adain else None,
                    speaker=speaker if args.speaker_emb else None,
                    use_gst=args.gst,
                )

                # Cal Loss
                mel_loss, mel_postnet_loss, d_loss, f_loss, e_loss = Loss(
                    log_duration_output,
                    log_D,
                    f0_output,
                    f0,
                    energy_output,
                    energy,
                    mel_output,
                    mel_postnet_output,
                    mel_target,
                    ~src_mask,
                    ~mel_mask,
                )
                total_loss = mel_loss + mel_postnet_loss + d_loss + f_loss + e_loss

                # Logger
                t_l = total_loss.item()
                m_l = mel_loss.item()
                m_p_l = mel_postnet_loss.item()
                d_l = d_loss.item()
                f_l = f_loss.item()
                e_l = e_loss.item()

                # Backward
                total_loss = total_loss / hp.acc_steps
                total_loss.backward()
                if current_step % hp.acc_steps != 0:
                    continue
                if current_step > 900000:
                    break

                # Clipping gradients to avoid gradient explosion

                nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip_thresh)

                # Update weights
                scheduled_optim.step_and_update_lr()
                scheduled_optim.zero_grad()

                # Print
                if current_step % hp.log_step == 0:
                    Now = time.perf_counter()

                    str1 = "Epoch [{}/{}], Step [{}/{}]:".format(
                        epoch + 1, hp.epochs, current_step, total_step
                    )
                    str2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Duration Loss: {:.4f}, F0 Loss: {:.4f}, Energy Loss: {:.4f};".format(
                        t_l, m_l, m_p_l, d_l, f_l, e_l
                    )
                    str3 = (
                        "Time Used: {:.3f}s, Estimated Time Remaining: {:.3f}s.".format(
                            (Now - Start), (total_step - current_step) * np.mean(Time)
                        )
                    )

                    print("\n" + str1)
                    print(str2)
                    print(str3)

                    with open(os.path.join(log_path, "log.txt"), "a") as f_log:
                        f_log.write(str1 + "\n")
                        f_log.write(str2 + "\n")
                        f_log.write(str3 + "\n")
                        f_log.write("\n")

                    train_logger.add_scalar("Loss/total_loss", t_l, current_step)
                    train_logger.add_scalar("Loss/mel_loss", m_l, current_step)
                    train_logger.add_scalar(
                        "Loss/mel_postnet_loss", m_p_l, current_step
                    )
                    train_logger.add_scalar("Loss/duration_loss", d_l, current_step)
                    train_logger.add_scalar("Loss/F0_loss", f_l, current_step)
                    train_logger.add_scalar("Loss/energy_loss", e_l, current_step)

                if current_step % hp.save_step == 0:
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        },
                        os.path.join(
                            checkpoint_path,
                            "checkpoint_{}.pth.tar".format(current_step),
                        ),
                    )
                    print("save model at step {} ...".format(current_step))

                if current_step % hp.synth_step == 0:
                    basename = id_[0]
                    src_length = src_len[0].item()
                    mel_length = mel_len[0].item()
                    mel_target = mel_target[0:1, :mel_length].detach().transpose(1, 2)
                    mel = mel_output[0:1, :mel_length].detach().transpose(1, 2)
                    mel_postnet = (
                        mel_postnet_output[0:1, :mel_length].detach().transpose(1, 2)
                    )

                    utils.vocoder_infer(
                        mel,
                        vocoder,
                        [
                            os.path.join(
                                hp.synth_path,
                                "step_{}_wo_postnet_{}.wav".format(
                                    current_step, basename
                                ),
                            )
                        ],
                    )
                    utils.vocoder_infer(
                        mel_postnet,
                        vocoder,
                        [
                            os.path.join(
                                hp.synth_path,
                                "step_{}_with_postnet_{}.wav".format(
                                    current_step, basename
                                ),
                            )
                        ],
                    )
                    utils.vocoder_infer(
                        mel_target,
                        vocoder,
                        [
                            os.path.join(
                                hp.synth_path,
                                "step_{}_reconstruct_{}.wav".format(
                                    current_step, basename
                                ),
                            )
                        ],
                    )

                    f0 = f0[0, :src_length].detach().cpu().numpy()
                    energy = energy[0, :src_length].detach().cpu().numpy()
                    f0_output = f0_output[0, :src_length].detach().cpu().numpy()
                    energy_output = energy_output[0, :src_length].detach().cpu().numpy()
                    duration = D[0, :src_length].detach().cpu().numpy().astype(np.int)

                    plot_mel(
                        [
                            (
                                mel_postnet[0].cpu().numpy(),
                                f0_output,
                                energy_output,
                                duration,
                            ),
                            (mel_target[0].cpu().numpy(), f0, energy, duration),
                        ],
                        ["Synthetized Spectrogram", "Ground-Truth Spectrogram"],
                        filename=os.path.join(
                            synth_path, "step_{}_{}.png".format(current_step, basename)
                        ),
                    )

                if len(Time) == hp.clear_Time:
                    Time = Time[:-1]
                end_time = time.perf_counter()
                Time = np.append(Time, end_time - start_time)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument("--d_vec", action="store_true")
    parser.add_argument("--x_vec", action="store_true")
    parser.add_argument("--adain", action="store_true")
    parser.add_argument("--speaker_emb", action="store_true")
    parser.add_argument("--gst", action="store_true")

    args = parser.parse_args()

    main(args)
