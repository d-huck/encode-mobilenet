import argparse
import logging
import random

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import sigmoid
from torch.nn.functional import binary_cross_entropy as bce
import torchaudio
from sklearn.model_selection import train_test_split
from torch.nn import init
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import MultilabelAveragePrecision
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import wandb
from data import (
    AudioSetDataset,
    AudioSetEpoch,
    AudioSetValidate,
    GTZANDataset,
    get_files,
)
from encodec import EncodecModel
from encodec.quantization import ResidualVectorQuantizer
from encodec.utils import convert_audio
from mobilenet import MobileNet, MobileNetV3_Smol

device = "cuda" if torch.cuda.is_available() else "cpu"

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(levelname)s :: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def train_one_epoch(
    model, optimizer, criterion, scheduler, train_data, valid_loader, epoch, pbar, args
):
    """Trains a single epoch of the model, Expects $n$ data items in the dataset,
    and performs it's own validation split on the data. A common strategy on AudioSet
    is to consider an epoch to be 100,000 samples. Here we will be doing a pseudo
    cross validation since we are splitting _after_ the data has been shuffled one time.
    Once we have gone through the entire dataset one time, we will past training examples
    in the valid set. We are not currently using he validation map or loss as any kind
    of signal as of yet.

    :param model: model to be trained
    :type model: nn.Module
    :param optimizer: optimizer
    :type optimizer: _type_
    :param criterion: loss function
    :type criterion: _type_
    :param scheduler: loss scheduler for the optimizer
    :type scheduler: _type_
    :param data: data to train the epoch
    :param epoch: which epoch is currently being trained
    :type epoch: int
    """

    metric = MultilabelAveragePrecision(num_labels=args.n_classes)

    train_set = AudioSetEpoch(train_data, device=args.device)
    train_loader = DataLoader(train_set, batch_size=args.batch_size)

    model.train()
    preds, targets = [], []
    t_loss, v_loss = 0, 0
    for x, y, z in tqdm(
        train_loader, leave=False, desc=f"Epoch: {epoch+1:03d} | Training  ", position=0
    ):
        optimizer.zero_grad()
        x, y, z = x.to(args.device), y.to(args.device), z.to(args.device)
        out = model(x)

        class_loss = criterion(out, y)

        out_soft = sigmoid(out / args.temperature)
        logits_soft = sigmoid(z / args.temperature)
        kd_loss = bce(out_soft, logits_soft)
        loss = (class_loss * args.kd_weight) + (kd_loss * (1 - args.kd_weight))
        # collect some metrics
        t_loss += loss.item()
        preds.append(out.detach().to("cpu"))
        targets.append(y.detach().to("cpu"))

        # do the thing
        loss.backward()
        optimizer.step()
        scheduler.step()
        pbar.update(1)

    preds = torch.cat(preds).to(args.device)
    targets = torch.cat(targets).type(torch.int32).to(args.device)
    train_map = metric(preds, targets).item()
    del preds, targets

    preds, targets = [], []
    model.eval()
    with torch.no_grad():
        for x, y, z in tqdm(
            valid_loader,
            leave=False,
            desc=f"Epoch: {epoch+1:03d} | Validation",
            position=0,
        ):
            x, y, z = x.to(args.device), y.to(args.device), z.to(args.device)
            out = model(x)
            class_loss = criterion(out, y)

            out_soft = sigmoid(out / args.temperature)
            logits_soft = sigmoid(z / args.temperature)
            kd_loss = bce(out_soft, logits_soft)
            loss = (class_loss * args.kd_weight) + (kd_loss * (1 - args.kd_weight))

            # collect some metrics
            v_loss += loss.item()
            preds.append(out.to("cpu"))
            targets.append(y.to("cpu"))
            pbar.update(1)

    preds = torch.cat(preds).to(args.device)
    targets = torch.cat(targets).type(torch.int32).to(args.device)
    valid_map = metric(preds, targets).item()
    del preds, targets

    tqdm.write(
        f"Epoch: {epoch+1:03d} | TL: {t_loss:.6f} | VL: {v_loss:.6f} | TmAP: {train_map:.4f} | VmAP: {valid_map:.4f}  | LR: {scheduler.get_last_lr()[0]:.4E}"
    )
    if args.wandb:
        wandb.log(
            {
                "train_loss": t_loss,
                "valid_loss": v_loss,
                "train_map": train_map,
                "valid_map": valid_map,
                "lr": scheduler.get_last_lr()[0],
            }
        )


def main(args):
    logger.info("Loading model")
    if args.model_size == "large":
        model = MobileNet(
            num_classes=args.n_classes, encodec_bw=args.encodec_bw, a=args.alpha
        )
    else:
        model = MobileNetV3_Smol(num_classes=args.n_classes, encodec_bw=args.encodec_bw)

    train_iter_per_epoch = int(np.ceil(args.examples_per_epoch / args.batch_size))

    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    sched_pct = args.warmup / args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        epochs=args.epochs,
        steps_per_epoch=train_iter_per_epoch,
        pct_start=sched_pct,
        max_lr=args.lr,
    )

    # load data
    logger.info("Loading data")

    dataset = AudioSetDataset(args.data_path, device=args.device)
    n_examples = len(dataset)
    epoch_loader = DataLoader(
        dataset,
        batch_size=args.examples_per_epoch,
        shuffle=True,
        drop_last=True,
    )

    valid_files = [file for file in get_files(args.valid_path, ext=".pt")]
    print(len(valid_files))
    valid_set = AudioSetValidate(valid_files, device=args.device)
    valid_loader = DataLoader(
        valid_set,
        num_workers=0,
        shuffle=False,
        batch_size=args.batch_size,
        pin_memory=True,
    )

    valid_iter_per_epoch = len(valid_loader)

    total_iter = args.epochs * (train_iter_per_epoch + valid_iter_per_epoch)
    logger.info("Starting training...")
    logger.info(
        f"Training for {total_iter} iterations with {train_iter_per_epoch} train and {valid_iter_per_epoch} validation iterations per epoch"
    )

    count = 0
    with tqdm(
        total=total_iter, desc="Total Training", position=1, smoothing=0.01
    ) as pbar:
        while count < args.epochs:
            for data in epoch_loader:
                train_one_epoch(
                    model,
                    optimizer,
                    criterion,
                    scheduler,
                    data,
                    valid_loader,
                    count,
                    pbar,
                    args,
                )
                count += 1
                if count >= args.epochs:
                    break
                if count % args.checkpoint_interval == 0:
                    checkpoint, ext = (
                        ".".join(args.checkpoint.split(".")[:-1]),
                        args.checkpoint.split(".")[-1],
                    )
                    checkpoint = f"{checkpoint}-e{count:03d}.{ext}"
                    torch.save(model.state_dict(), checkpoint)

        torch.save(model.state_dict(), args.checkpoint)


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
        "--model_size",
        type=str,
        default="large",
        choices=["large", "small"],
        help="Which model size to use",
    )
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
        "--valid_size",
        type=float,
        default=0.1,
        help="Validation set ratio for each epoch",
    )
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
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for knowledge distillation",
    )
    parser.add_argument(
        "--kd_weight",
        type=float,
        default=1.0,
        help="Lambda for knowledge distillation. Higher values will increase weight of classication loss",
    )
    parser.add_argument(
        "--target_device", type=int, default=0, help="Sets the target device"
    )
    args = parser.parse_args()

    if args.wandb:
        wandb.init(project="encodec-mobilenet-as")
        wandb.config.update(args)

    if args.device == "cuda":
        torch.cuda.set_device(args.target_device)
        args.device = f"cuda:{args.target_device}"

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    main(args)
