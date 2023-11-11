import argparse
import json
import os
import random
import time
from copy import deepcopy

import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

import multiprocessing as mp

from encodec import EncodecModel
from encodec.utils import convert_audio


def get_files(fp: str) -> list:
    for root, dirr, files in os.walk(fp):
        for f in files:
            if f.endswith(".flac"):
                yield os.path.join(root, f)


def load_meta(fp: str) -> dict:
    meta = pd.read_csv(
        fp, skiprows=3, sep=", ", names=["YTID", "start", "dur", "labels"]
    )
    return pd.Series(meta.labels.values, index=meta.YTID).to_dict()


def load_labels(fp: str) -> dict:
    return json.load(open(fp, "r"))["mid2int"]


def encode_data(data, encoder, batch_size=16, device=None):
    """Encodes data to the target bandwidth using Encodec

    :param data: list of audio items
    :type data: _type_
    :param encoder: Encodec Model
    :type encoder: _type_
    :param batch_size: Batch size for inference, defaults to 32
    :type batch_size: int, optional
    :param device: torch device to use, defaults to None
    :type device: _type_, optional
    :return: encoded data
    :rtype: _type_
    """
    encodings = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = torch.cat(data[i : i + batch_size], dim=0).to(device)
            encoded_frames = encoder.encode(batch)

            codes = torch.cat([e[0] for e in encoded_frames], dim=-1)
            codes = codes.to("cpu").detach()
            for code in torch.split(codes, 1, dim=0):
                encodings.append(code.clone())

    return encodings


def make_dataset(
    fp: str,
    meta: dict,
    encoder: EncodecModel,
    mid2int: dict,
    target_sr=1.5,
    batch_size=1024,
    device="cpu",
):
    """Creates a dataset in memory from a list of flac files and the appropriate metadata.

    :param fp: path of folder holding flac files
    :type fp: str
    :param meta: dictionary holding the metadata of the dataset
    :type meta: dict
    :param encoder: Encodec model to use for encoding
    :type encoder: EncodecModel
    :param mid2int: dictionary to translate the label string to integers
    :type mid2int: dict
    :param target_sr: target sample rate for the encoder, defaults to 1.5
    :type target_sr: float, optional
    :param batch_size: how many samples to collect before encoding, defaults to 1024
    :type batch_size: int, optional
    :param device: torch device, defaults to cpu
    :type device: _type_, optional
    """
    out = {"data": [], "targets": []}
    batch = []

    max_len = encoder.sample_rate * 10
    files = get_files(fp)
    for f in tqdm(files, smoothing=0.1):
        ytid = os.path.basename(f).split(".")[0]
        if ytid not in meta:
            continue
        audio, sr = torchaudio.load(f)
        audio = convert_audio(audio, sr, encoder.sample_rate, encoder.channels)
        if audio.shape[-1] < max_len:
            audio = F.pad(audio, (0, max_len - audio.shape[-1]))
        elif audio.shape[-1] > max_len:
            audio = audio.narrow(-1, 0, max_len)

        targets = meta[ytid].replace('"', "").split(",")
        targets = [mid2int[x] for x in targets]
        batch.append(audio.unsqueeze(0))
        if len(batch) == batch_size:
            encoded = encode_data(batch, encoder)
            out["data"].append(encoded.to("cpu"))
            batch = []
            torch.cuda.empty_cache()
        out["targets"].append(targets)

    # need to check for remaining items to be encoded
    if len(batch) > 0:
        encoded = encode_data(batch, encoder)
        out["data"].append(encoded.to("cpu"))
        torch.cuda.empty_cache()

    out["data"] = torch.cat(out["data"], dim=0)
    return out


def load_worker(file_q, meta, label_dict, encode_q, write_q, target_sr, channels, done):
    max_len = target_sr * 10
    while True:
        fp = file_q.get()
        if fp is None:
            encode_q.put(None)
            break
        ytid = os.path.basename(fp).split(".")[0]
        if ytid not in meta:
            continue
        audio, sr = torchaudio.load(fp)
        audio = convert_audio(audio, sr, target_sr, channels)
        if audio.shape[-1] < max_len:
            audio = F.pad(audio, (0, max_len - audio.shape[-1]))
        elif audio.shape[-1] > max_len:
            audio = audio.narrow(-1, 0, max_len)

        labels = meta[ytid].replace('"', "").split(",")
        labels = [label_dict[x] for x in labels]
        data = {"audio": audio.unsqueeze(0), "labels": labels, "ytid": ytid}
        encode_q.put(data)
    done.wait()


def encode_worker(encode_q, write_q, done, args, batch_size=128):
    batch = []
    none_counter = 0
    encoder = EncodecModel.encodec_model_24khz()
    encoder.set_target_bandwidth(args.target_br)
    encoder.to(args.device)

    while True:
        data = encode_q.get()
        if data is None:
            none_counter += 1
            if none_counter == args.num_workers:
                break
            continue
        batch.append(deepcopy(data))
        if len(batch) >= batch_size:
            audios = [x["audio"] for x in batch]
            out = encode_data(audios, encoder, batch_size=128, device=args.device)

            for i, d in enumerate(batch):
                d["audio"] = out[i]
                write_q.put(d)
            batch = []

    # take care of the stragglers
    if len(batch) > 0:
        audios = torch.cat([x["audio"] for x in batch], dim=0)
        encoded = encode_data(audios, encoder, batch_size=128).cpu()
        for i, d in enumerate(batch):
            d["audio"] = encoded[i].clone().detach()
            write_q.put(d)
    write_q.put(None)
    done.wait()


def q_monitor(file_q, encode_q, write_q, done):
    while True:
        if file_q.qsize() == 0 and encode_q.qsize() == 0 and write_q.qsize() == 0:
            break
        print(
            f"File Queue: {file_q.qsize()} | Encode Queue: {encode_q.qsize()} | Write Queue: {write_q.qsize()}"
        )
        time.sleep(5)

    done.wait()


def write_worker(write_q, fp, done):
    while True:
        data = write_q.get()
        if data is None:
            break
        data = deepcopy(data)
        # write to disk
        torch.save(data, os.path.join(fp, f"{data['ytid']}.pt"))

    done.set()


def main(args):
    # build the queues
    file_q = mp.Queue()
    encoder_q = mp.Queue()
    write_q = mp.Queue()
    done = mp.Event()

    meta = load_meta(args.meta)
    label_dict = load_labels(args.labels)

    workers = []
    encoded = [os.path.basename(x).split(".")[0] for x in get_files(args.output_dir)]

    for i, file in enumerate(get_files(args.input_dir)):
        ytid = os.path.basename(file).split(".")[0]
        if ytid in encoded:
            continue
        file_q.put(file)

    workers.append(
        mp.Process(
            target=q_monitor,
            args=(file_q, encoder_q, write_q, done),
        )
    )

    for i in range(args.num_workers):
        file_q.put(None)
        workers.append(
            mp.Process(
                target=load_worker,
                args=(
                    file_q,
                    meta,
                    label_dict,
                    encoder_q,
                    write_q,
                    args.target_sr,
                    args.channels,
                    done,
                ),
            )
        )

    workers.append(
        mp.Process(
            target=encode_worker,
            args=(encoder_q, write_q, done, args),
        )
    )

    workers.append(
        mp.Process(
            target=write_worker,
            args=(write_q, args.output_dir, done),
        )
    )

    for worker in workers:
        worker.start()

    done.wait()
    for worker in workers:
        worker.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--input_dir", type=str, required=True, help="Path to the input directory"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory",
    )
    parser.add_argument(
        "--meta", type=str, required=True, help="Path to the metadata file"
    )
    parser.add_argument(
        "--labels", type=str, required=True, help="Path to the labels file"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of torch load workers"
    )
    parser.add_argument(
        "--target_br", type=float, default=1.5, help="Target sample rate for encoder"
    )
    parser.add_argument(
        "--channels", type=int, default=1, help="Number of channels for encoder"
    )
    parser.add_argument(
        "--target_sr", type=int, default=24_000, help="Target sample rate for encoder"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="torch device to use",
    )

    args = parser.parse_args()
    torch.multiprocessing.set_start_method("spawn")
    torch.multiprocessing.set_sharing_strategy("file_system")

    main(args)
