import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss
from dataset import Dataset

from evaluate import evaluate


import aim
from aim.pytorch import track_params_dists, track_gradients_dists
import numpy as np
from track_utils import fig_to_img, track_model_graph
import matplotlib.pyplot as plt
from chart_studio import plotly
import chart_studio.plotly as py
import plotly.tools as tls
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def main(args, configs):
	print("Prepare training ...")

	preprocess_config, model_config, train_config = configs

	# Get dataset
	dataset = Dataset(
		"train.txt", preprocess_config, train_config, sort=True, drop_last=True
	)
	batch_size = train_config["optimizer"]["batch_size"]
	group_size = 4  # Set this larger than 1 to enable sorting in Dataset
	assert batch_size * group_size < len(dataset)
	loader = DataLoader(
		dataset,
		batch_size=batch_size * group_size,
		shuffle=True,
		collate_fn=dataset.collate_fn,
	)

	# Prepare model
	model, optimizer = get_model(args, configs, device, train=True)
	model = nn.DataParallel(model)
	num_param = get_param_num(model)
	Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
	print("Number of FastSpeech2 Parameters:", num_param)

	# Load vocoder
	vocoder = get_vocoder(model_config, device)

	# Init logger
	for p in train_config["path"].values():
		os.makedirs(p, exist_ok=True)
	train_log_path = os.path.join(train_config["path"]["log_path"], "train")
	val_log_path = os.path.join(train_config["path"]["log_path"], "val")
	os.makedirs(train_log_path, exist_ok=True)
	os.makedirs(val_log_path, exist_ok=True)
	train_logger = None #SummaryWriter(train_log_path)
	val_logger = None #SummaryWriter(val_log_path)

	# Training
	step = args.restore_step + 1
	epoch = 1
	grad_acc_step = train_config["optimizer"]["grad_acc_step"]
	grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
	total_step = train_config["step"]["total_step"]
	log_step = train_config["step"]["log_step"]
	save_step = train_config["step"]["save_step"]
	synth_step = train_config["step"]["synth_step"]
	val_step = train_config["step"]["val_step"]

	outer_bar = tqdm(total=total_step, desc="Training", position=0)
	outer_bar.n = args.restore_step
	outer_bar.update()


	experiment_name = "FS2"
	if args.aim_server is not None:
		remote_tracking_server = f'aim://{args.aim_server}'
		aim_run  = aim.Run(experiment = experiment_name, repo = remote_tracking_server)
	else:
		aim_run  = aim.Run(experiment = experiment_name)

	aim_run["train_config"] = train_config
	aim_run["preprocess_config"] = preprocess_config
	# aim_run["model_graph_metadata"] = track_model_graph(model, )
	metadata_graph_metadata = track_model_graph(model)
	aim_run["metadata_graph_metadata"] = metadata_graph_metadata
	while True:
		inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
		for batchs in loader:
			for batch in batchs:
				batch = to_device(batch, device)

				# Forward
				output = model(*(batch[2:]))

				# Cal Loss
				losses = Loss(batch, output)
				total_loss = losses[0]

				# Backward
				total_loss = total_loss / grad_acc_step

				total_loss.backward()


				if step % grad_acc_step == 0:
					# Clipping gradients to avoid gradient explosion

					grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

					if math.isnan(grad_norm):
						print("grad_norm is nan. Not Updating.")
					else:
						optimizer.step_and_update_lr()
					optimizer.zero_grad()

				if step % log_step == 0:

					total_loss,mel_loss, postnet_mel_loss,pitch_loss,energy_loss,duration_loss = losses

					aim_run.track(total_loss.item() , name = "Loss", context = {'type':'total_loss'})
					aim_run.track(mel_loss.item() , name = "Loss", context = {'type':'mel_loss'})
					aim_run.track(postnet_mel_loss.item() , name = "Loss", context = {'type':'postnet_mel_loss'})
					aim_run.track(energy_loss.item() , name = "Loss", context = {'type':'pitch_loss'})
					aim_run.track(duration_loss.item() , name = "Loss", context = {'type':'duration_loss'})

					losses = [l.item() for l in losses]

					message1 = "Step {}/{}, ".format(step, total_step)
					message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
						*losses
					)

					track_params_dists(model, aim_run)
					track_gradients_dists(model, aim_run)

					aim_run.track(aim.Text(message1 + message2 + "\n"), name = 'log_out')

					with open(os.path.join(train_log_path, "log.txt"), "a") as f:
						f.write(message1 + message2 + "\n")

					outer_bar.write(message1 + message2)

					# log(train_logger, step, losses=losses)

				if step % synth_step == 0:
					fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
						batch,
						output,
						vocoder,
						model_config,
						preprocess_config,
					)

					aim_run.track([
						aim.Audio(wav_reconstruction, format='wav', caption = 'ground truth'),
						aim.Audio(wav_prediction, format='wav', caption = 'pred')
					], name = 'waves', context = {'type':'waves_pred_gt'})
					# aim_run.track(aim.Audio(wav_prediction, format='wav'), name = 'waves',  context = {'type':'wav_prediction'})

					plotly_fig = tls.mpl_to_plotly(fig)

					aim_run.track(aim.Image(fig_to_img(fig)), name = 'Sepctrograms',  context = {'type':'MEL'})
					aim_run.track(aim.Figure(plotly_fig), name = 'Sepctrograms',  context = {'type':'MEL Interactive'})


				if step % val_step == 0:
					model.eval()
					message = evaluate(model, step, configs, val_logger, vocoder)
					with open(os.path.join(val_log_path, "log.txt"), "a") as f:
						f.write(message + "\n")
					outer_bar.write(message)

					model.train()

				if step % save_step == 0:
					torch.save(
						{
							"model": model.module.state_dict(),
							"optimizer": optimizer._optimizer.state_dict(),
						},
						os.path.join(
							train_config["path"]["ckpt_path"],
							"{}.pth.tar".format(step),
						),
					)

				if step == total_step:
					quit()
				step += 1
				outer_bar.update(1)

			inner_bar.update(1)
		epoch += 1





if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--restore_step", type=int, default=0)
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
			"-as", "--aim_server", type=str, default=None,required=False, help="Remote aim server ip:port"
		)
	args = parser.parse_args()

	# Read Config
	preprocess_config = yaml.load(
		open(args.preprocess_config, "r"), Loader=yaml.FullLoader
	)
	model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
	train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
	configs = (preprocess_config, model_config, train_config)

	main(args, configs)
