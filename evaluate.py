import argparse
import os
import re

import numpy as np
import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder
from utils.tools import (
    create_spe_stats_fig,
    to_device,
    log,
    synth_one_sample,
    create_phone_batch,
)
from model import FastSpeech2Loss
from dataset import Dataset

from text import text_to_sequence, sequence_to_text

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(
    model,
    step,
    configs,
    logger=None,
    vocoder=None,
    spe_classifier=None,
    save_figure=False,
    val_utts="val.txt",
    mfcc=False,
    feature_wise_loss=False,
):
    preprocess_config, model_config, train_config = configs
    use_energy = model_config["variance_predictor"]["use_energy_predictor"]

    if "val_file" in train_config["path"] and train_config["path"]["val_file"]:
        val_utts = train_config["path"]["val_file"]

    dataset = Dataset(
        val_utts, preprocess_config, train_config, sort=False, drop_last=False
    )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )
    if train_config["path"]["spe_classifier_ckpt"]:
        from spe_classifier.utils import collect_frames, pad_2D, FEAT_LABELS, two_bit_to_spe

    # Get loss function
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)

    # Evaluation
    loss_sums = [0 for _ in range(6)]
    h0 = []
    h3 = []
    accs = []
    recalls = []
    precisions = []
    for batchs in loader:
        for batch in batchs:
            batch = to_device(batch, device)
            with torch.no_grad():
                # Forward
                output = model(*(batch[2:]))

                # Cal Loss
                losses = Loss(batch, output)

                for i in range(len(losses)):
                    loss_sums[i] += losses[i].item() * len(batch[0])

            if train_config["path"]["spe_classifier_ckpt"]:
                from spe_classifier.utils import hamming_distance

                spe_classifier.eval()
                WIDTH = 15
                pb_mels = create_phone_batch(output)
                pb_mels = [
                    collect_frames(phone, WIDTH) for utt in pb_mels for phone in utt
                ]
                pb_mels = pad_2D(pb_mels, WIDTH)
                pb_mels = torch.from_numpy(pb_mels).float().to(device)
                with torch.no_grad():
                    pb_out = spe_classifier(pb_mels).squeeze(0)  # (592, 72)

                if model_config["transformer"]["spe_features"]:
                    phone_labels = (
                        batch[4]
                        .reshape(
                            batch[4].shape[0] * batch[4].shape[1], batch[4].shape[2]
                        )
                        .cpu()
                        .numpy()
                    )
                else:
                    # If original model was character based, recreate the features
                    # TODO: sort this out for multilingual character based models
                    phone_labels = re.sub(
                        "(OW|AW|AY|EY|OY)\d",
                        "_",
                        "{"
                        + " ".join(
                            sequence_to_text(
                                batch[4]
                                .reshape(batch[4].shape[0] * batch[4].shape[1])
                                .cpu()
                                .numpy(),
                                preprocess_config["preprocessing"]["text"]["language"],
                            )
                        )
                        + "}",
                    )
                    one_hots = np.array(
                        text_to_sequence(
                            phone_labels,
                            False,
                            preprocess_config["preprocessing"]["text"]["language"],
                        )
                    )
                    phone_labels = np.array(
                        text_to_sequence(
                            phone_labels,
                            True,
                            preprocess_config["preprocessing"]["text"]["language"],
                        )
                    )
                # phone_labels = phone_labels[np.where(np.any(phone_labels > 0, axis=1))] # remove padding
                whole_output = pb_out.cpu().numpy()
                whole_output = np.round(np.where(whole_output <= 0.5, 0, 1), 0).astype(
                    int
                )
                whole_output = np.array([two_bit_to_spe(list(x)) for x in whole_output])
                # whole_output = whole_output[np.where(np.any(phone_labels>0, axis=1))]
                hamming_scores = np.array(
                    [
                        model_config["transformer"]["spe_feature_dim"]
                        - sum(whole_output[i] == phone_labels[i])
                        for i in range(len(whole_output))
                    ]
                )
                hamming_scores_no_pad = hamming_scores[
                    np.where(np.any(phone_labels > 0, axis=1))
                ]
                hamming0 = hamming_scores_no_pad[np.where(hamming_scores_no_pad == 0)]
                hamming0 = len(hamming0) / len(hamming_scores_no_pad)
                hamming3 = hamming_scores_no_pad[np.where(hamming_scores_no_pad < 4)]
                hamming3 = len(hamming3) / len(hamming_scores_no_pad)
                # hamming3 = hamming_distance(whole_output, phone_labels, 3)
                # TODO: replace in main SPE classifier repo
                h0.append(hamming0)
                h3.append(hamming3)
                no_pad_output = whole_output[np.where(np.any(phone_labels > 0, axis=1))]
                no_pad_labels = phone_labels[np.where(np.any(phone_labels > 0, axis=1))]
                flipped = np.swapaxes(no_pad_output, 0, 1)
                flipped_labs = np.swapaxes(no_pad_labels, 0, 1)
                acc = (
                    np.array(
                        [
                            np.sum(flipped[i] == flipped_labs[i])
                            for i in range(flipped.shape[0])
                        ]
                    )
                    / flipped.shape[1]
                )
                n_labels = len(FEAT_LABELS)
                accs.append(acc[:n_labels])
                # precision, recall, accuracy = calculate_stats(flipped, flipped_labs)
                # n_labels = len(FEAT_LABELS)
                # accs.append(accuracy[:n_labels])
                # recalls.append(recall[:n_labels])
                # precisions.append(precision[:n_labels])

    loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]
    if train_config["path"]["spe_classifier_ckpt"]:
        h0_avg = sum(h0) / len(h0)
        h3_avg = sum(h3) / len(h3)
        voi_avg = np.mean([x[8] for x in accs])
        # precision_avg = np.mean([v for x in precisions for v in x if v])
        # recall_avg = np.mean([v for x in recalls for v in x if v])
        acc_avg = np.mean([v for x in accs for v in x if v])
        feats = np.round(np.mean(accs, axis=0) * 100, 2)
        # recall_feats = np.round(np.mean(recalls, axis=0) * 100, 2)
        # prec_feats = np.round(np.mean(precisions, axis=0) * 100, 2)

        spe_fig = create_spe_stats_fig(feats, save_figure=save_figure)
        if logger is not None:
            log(
                logger,
                fig=spe_fig,
                tag="Validation/spe_classifier_step_{}".format(step),
                use_energy=use_energy,
            )
            logger.add_scalar("Validation/spe_accuracy", acc_avg, step)
            # logger.add_scalar("Validation/spe_recall", recall_avg, step)
            # logger.add_scalar("Validation/spe_precision", precision_avg, step)

        message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}, Avg Hamming Distance (0): {:.4f}%,  Avg Hamming Distance (3): {:.4f}%, Avg SPE Classifier Accuracy: {:.4f}%, Voice feature Average Acc: {:.4f}%".format(
            *(
                [step]
                + [l for l in loss_means]
                + [
                    h0_avg * 100,
                    h3_avg * 100,
                    acc_avg * 100,
                    # precision_avg * 100,
                    # recall_avg * 100,
                    voi_avg * 100,
                ]
            )
        )
    else:
        message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
            *([step] + [l for l in loss_means])
        )

    if logger is not None:
        fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
            batch,
            output,
            vocoder,
            model_config,
            preprocess_config,
        )
        # log here?
        log(logger, step, losses=loss_means, use_energy=use_energy)
        log(
            logger,
            fig=fig,
            tag="Validation/step_{}_{}".format(step, tag),
            use_energy=use_energy,
        )
        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        log(
            logger,
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_reconstructed".format(step, tag),
            use_energy=use_energy,
        )
        log(
            logger,
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_synthesized".format(step, tag),
            use_energy=use_energy,
        )

    return message


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=30000)
    parser.add_argument(
        "-q", "--quick_config", type=str, required=False, help="config slug"
    )
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=False,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=False, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=False, help="path to train.yaml"
    )
    parser.add_argument(
        "-v", "--val_utts", type=str, required=False, help="path to val utts"
    )
    args = parser.parse_args()

    # Read Config
    if args.quick_config:
        # Read Config
        preprocess_config = yaml.load(
            open(f"config/{args.quick_config}/preprocess.yaml", "r"),
            Loader=yaml.FullLoader,
        )
        model_config = yaml.load(
            open(f"config/{args.quick_config}/model.yaml", "r"), Loader=yaml.FullLoader
        )
        train_config = yaml.load(
            open(f"config/{args.quick_config}/train.yaml", "r"), Loader=yaml.FullLoader
        )
    else:
        preprocess_config = yaml.load(
            open(args.preprocess_config, "r"), Loader=yaml.FullLoader
        )
        model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
        train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False).to(device)
    spe_classifier = None
    if train_config["path"]["spe_classifier_ckpt"]:
        from spe_classifier.inference import get_model as get_spe_classifier

        spe_classifier = get_spe_classifier(train_config["path"]["spe_classifier_ckpt"])

    if args.val_utts:
        utts = args.val_utts
    else:
        utts = "val.txt"

    message = evaluate(
        model,
        args.restore_step,
        configs,
        spe_classifier=spe_classifier,
        save_figure=True,
        val_utts=utts,
    )
    print(message)
