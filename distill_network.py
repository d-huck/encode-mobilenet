"""Distill MobileNet from PaSST and/or AST. Training examples should have the form

{
    ['encodec']: <encodec indices>,
    ['passt']: PaSST logits. If logits are from ensemble, they should be averaged offline,
    ['ast']: AST logits. Same as above,
    ['labels']: AudioSet labels as indices for one hot vector creation
}

"""

import argparse
import logging
import random

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import lightning as pl
import wandb


def main():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f",
        "--data_path",
        type=str,
        default="data",
        required=True,
        help="Directory holding training examples",
    )
    parser.add_argument(
        "-o",
        "--checkpoint",
        type=str,
        default="output.ptw",
        help="Path to save the model",
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument(
        "--epochs", type=int, default=100, help="Total number of epochs to train for"
    )
    parser.add_argument(
        "--examples_per_epoch",
        type=int,
        default=100_000,
        help="Number of examples we will use in each epoch",
    ),
    parser.add_argument(
        "--lr",
        type=float,
        default=8e-4,
        help="The max learning rate of the optimizer. This is used in the OneCycleLR scheduler.",
    ),
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Use mixed precision training",
    ),
    parser.add_argument(
        "--n_classes", type=int, default=527, help="Number of classes in the dataset"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Which device to use",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--warmup",
        type=int,
        default=8,
        help="Number of epochs to warmup the learning rate",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=10,
        help="how_often to save the model. Unimplemented",
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of torch load workers"
    )
    parser.add_argument(
        "--wandb", action="store_true", default=False, help="Use wandb for logging"
    )
    parser.add_argument(
        "--encodec_bw", type=float, default=12.0, help="Target bandwidth for encodec"
    )

    parser.add_argument(
        "--valid_path", type=str, default="data/valid", help="Path to validation data"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Sets the width of the model. alpha == 1 produces MobileNetV3-Large, while any other value scales the width of the model.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.33,
        help="Scales the distillation loss from PaSST. Used in conjunction with gamma and lambda.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.33,
        help="Scales the distillation loss from AST. Used in conjunction with beta and lambda.",
    )
    parser.add_argument(
        "--lambda",
        type=float,
        default=0.34,
        help="Scales the BCE classification loss from the base model. Used in conjunction with beta and gamma.",
    )

    args = parser.parse_args()

    if args.wandb:
        wandb.init(project="encodec-mobilenet-as")
        wandb.config.update(args)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    main(args)
