"""
Runs through the dataset and finds any files which fail to load
"""
import argparse
from ast import arg
import torch
from torch.utils.data import DataLoader
from data import AudioSetDataset, AudioSetEpoch, get_files
from tqdm import tqdm


def main(args):
    train = AudioSetEpoch([x for x in get_files(args.train, ext=".pt")])
    train = DataLoader(
        train,
        batch_size=1024,
        shuffle=False,
        num_workers=8,
    )
    for t in tqdm(train, leave=False, desc="Training Files"):
        x, y = t

    val = AudioSetEpoch([x for x in get_files(args.val, ext=".pt")])
    val = DataLoader(
        val,
        batch_size=1024,
        shuffle=False,
        num_workers=0,
    )
    for v in tqdm(val, leave=False, desc="Validation Files"):
        x, y = t

    test = AudioSetEpoch([x for x in get_files(args.test, ext=".pt")])
    test = DataLoader(
        test,
        batch_size=1024,
        shuffle=False,
        num_workers=0,
    )
    for t in tqdm(test, leave=False, desc="Test Files"):
        x, y = t


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        type=str,
        required=True,
        help="Path to the training data directory",
    )
    parser.add_argument(
        "--val",
        type=str,
        required=True,
        help="Path to the validation data directory",
    )
    parser.add_argument(
        "--test",
        type=str,
        required=True,
        help="Path to the test data directory",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of torch load workers",
    )

    args = parser.parse_args()
    main(args)
