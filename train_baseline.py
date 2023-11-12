import argparse
import logging
from ast import parse

import torch
import torch.nn as nn
import torchaudio
from sklearn.model_selection import train_test_split
from torch.nn import init
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import MultilabelAveragePrecision
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from data import AudioSetDataset, AudioSetEpoch, GTZANDataset, split_data
from encodec import EncodecModel
from encodec.quantization import ResidualVectorQuantizer
from encodec.utils import convert_audio
from mobilenet import MobileNetV3_LARGE, MobileNetV3_Smol

device = "cuda" if torch.cuda.is_available() else "cpu"

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(levelname)s :: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def train_one_epoch(model, optimizer, criterion, scheduler, data, epoch, pbar, args):
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

    train, valid = split_data(
        data,
        batch_size=args.batch_size,
        random_seed=args.seed,
        device=args.device,
        test_size=args.valid_size,
        dataset_t=AudioSetEpoch,
    )

    model.train()
    preds, targets = [], []
    t_loss, v_loss = 0, 0
    for x, y in tqdm(
        train, leave=False, desc=f"Epoch: {epoch+1:03d} | Training  ", position=0
    ):
        optimizer.zero_grad()
        x, y = x.to(args.device), y.to(args.device)
        out = model(x)
        loss = criterion(out, y)

        # collect some metrics
        t_loss += loss.item()
        preds.append(out)
        targets.append(y)

        # do the thing
        loss.backward()
        optimizer.step()
        scheduler.step()
        pbar.update(1)

    preds = torch.cat(preds)
    targets = torch.cat(targets).type(torch.int32)
    train_map = metric(preds, targets).to("cpu").item()
    preds, targets = [], []

    model.eval()
    with torch.no_grad():
        for x, y in tqdm(
            valid, leave=False, desc=f"Epoch: {epoch+1:03d} | Validation", position=0
        ):
            x, y = x.to(args.device), y.to(args.device)
            out = model(x)
            loss = criterion(out, y)

            # collect some metrics
            v_loss += loss.item()
            preds.append(out)
            targets.append(y)
            pbar.update(1)

    preds = torch.cat(preds)
    targets = torch.cat(targets).type(torch.int32)
    valid_map = metric(preds, targets).to("cpu").item()

    tqdm.write(
        f"Epoch: {epoch+1:03d} | TL: {t_loss:.6f} | VL: {v_loss:.6f} | TmAP: {train_map:.4f} | VmAP: {valid_map:.4f}  | LR: {scheduler.get_last_lr()[0]:.4E}"
    )


def main(args):
    logger.info("Loading model")
    if args.model_size == "large":
        model = MobileNetV3_LARGE(num_classes=args.n_classes)
    else:
        model = MobileNetV3_Smol(num_classes=args.n_classes)

    train_iter_per_epoch = (
        int(args.examples_per_epoch * (1 - args.valid_size) / args.batch_size) + 1
    )
    valid_iter_per_epoch = (
        int(args.examples_per_epoch * args.valid_size / args.batch_size) + 1
    )

    total_iter = args.epochs * (train_iter_per_epoch + valid_iter_per_epoch)
    logger.info(
        f"Training for {total_iter} iterations with {train_iter_per_epoch} train and {valid_iter_per_epoch} validation iterations per epoch"
    )

    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    sched_pct = args.warmup / args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        epochs=args.epochs,
        steps_per_epoch=train_iter_per_epoch,
        pct_start=sched_pct,
        max_lr=1e-3,
    )

    # load data
    logger.info("Loading data")
    # dataset = torch.load("./audioset_encodings-12.0-unbalanced-330k.data")
    # n_examples = len(dataset["data"])

    dataset = AudioSetDataset(args.data_path, device=args.device)
    n_examples = len(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=args.examples_per_epoch,
        shuffle=False,
        drop_last=True,
        # num_workers=4,
    )

    count = 0
    logger.info("Starting training...")
    with tqdm(
        total=total_iter, desc="Total Training", position=1, smoothing=0.01
    ) as pbar:
        while count < args.epochs:
            for data in dataloader:
                train_one_epoch(
                    model, optimizer, criterion, scheduler, data, count, pbar, args
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
        default=1e-3,
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
        default=10,
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

    args = parser.parse_args()
    # if args.num_workers > 0:
    #     torch.multiprocessing.set_start_method("spawn")
    main(args)
