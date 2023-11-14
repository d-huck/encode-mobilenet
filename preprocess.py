"""

Pre processer for the entire dataset. This precomputes and caches the ast logits
as well as the Encodec codebook _indices_ and targets for the entire dataset. The 
incoming data should consist of flac audioset files as well as a dictionary of labels.
The output will look like this:

data = {
    "ytid": str,
    "audio_tokens": torch.Tensor [128, 750],
    "ast_logits": torch.Tensor [527],
    "labels": list ]
}

and should be stored as {ytid}.pt
"""

import argparse
from json import encoder
import os
from re import A
import time
from uu import encode
from copy import deepcopy
import encodec
from create_encoded_audioset import write_worker
from data import get_files
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import torch
import torch.nn.functional as F
import torchaudio
import multiprocessing as mp
from torchaudio.transforms import Resample
from encodec import EncodecModel
from encodec.utils import convert_audio
from transformers import AutoFeatureExtractor, ASTForAudioClassification

AST_SR = 16000
ENCODEC_SR = 24000
MAX_LEN = 10
N_CHANNELS = 1


def load_meta(fp: str) -> dict:
    meta = pd.read_csv(
        fp,
        skiprows=3,
        sep=", ",
        names=["YTID", "start", "dur", "labels"],
        engine="python",
    )
    return pd.Series(meta.labels.values, index=meta.YTID).to_dict()


def load_labels(fp: str) -> dict:
    return json.load(open(fp, "r"))["mid2int"]


def encode_data(batch, encoder, args):
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
    batch = batch.to(args.device)
    with torch.no_grad():
        encoded_frames = encoder.encode(batch)
        codes = torch.cat([e[0] for e in encoded_frames], dim=-1)
        codes = codes.to("cpu").detach()

    encodings = [code.clone() for code in torch.split(codes, 1, dim=0)]
    return encodings


def prepare_audio(fp, resampler, feature_extractor):
    audio, sr = torchaudio.load(fp)
    encodec_audio = convert_audio(audio, sr, ENCODEC_SR, N_CHANNELS).unsqueeze(0)

    ast_len = int(MAX_LEN * AST_SR)
    encodec_len = int(MAX_LEN * ENCODEC_SR)

    if encodec_audio.shape[-1] < encodec_len:
        encodec_audio = F.pad(encodec_audio, (0, encodec_len - encodec_audio.shape[-1]))
    elif encodec_audio.shape[-1] > encodec_len:
        encodec_audio = encodec_audio.narrow(-1, 0, encodec_len)

    ast_audio = resampler(audio)
    if ast_audio.shape[-1] < ast_len:
        ast_audio = F.pad(ast_audio, (0, ast_len - ast_audio.shape[-1]))
    elif ast_audio.shape[-1] > ast_len:
        ast_audio = ast_audio.narrow(-1, 0, ast_len)

    return encodec_audio, ast_audio.squeeze()


def get_ast_logits(batch, model, args):
    inputs = batch.to(args.device)
    with torch.no_grad():
        logits = model(**inputs).logits
    logits = logits.to("cpu").detach()
    return [l.clone().squeeze() for l in torch.split(logits, 1, dim=0)]


def load_worker(file_q, meta, labels_dict, encode_q, args, done):
    resampler = Resample(
        orig_freq=48000,
        new_freq=AST_SR,
        resampling_method="sinc_interp_kaiser",
        lowpass_filter_width=64,
        rolloff=0.9475937167399596,
        beta=14.769656459379492,
    )

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    )
    batch = {
        "encodec_audio": [],
        "ast_audio": [],
        "labels": [],
        "ytid": [],
    }

    while True:
        fp = file_q.get()
        if fp is None:
            encode_q.put(None)
            break

        ytid = os.path.basename(fp).split(".")[0]
        if ytid not in meta:
            continue

        e, a = prepare_audio(fp, resampler, feature_extractor)
        labels = meta[ytid].replace('"', "").split(",")
        labels = [labels_dict[x] for x in labels]
        batch["encodec_audio"].append(e)
        batch["ast_audio"].append(a)
        batch["labels"].append(labels)
        batch["ytid"].append(ytid)

        if len(batch["encodec_audio"]) == args.batch_size:
            batch["encodec_audio"] = torch.cat(batch["encodec_audio"], dim=0)
            batch["ast_audio"] = feature_extractor(
                np.stack(batch["ast_audio"]), sampling_rate=AST_SR, return_tensors="pt"
            )
            encode_q.put(batch)
            batch = {
                "encodec_audio": [],
                "ast_audio": [],
                "labels": [],
                "ytid": [],
            }

    encode_q.put(batch)
    done.wait()


def encode_worker(encode_q, ast_q, done, args):
    none_ctr = 0
    encoder = EncodecModel.encodec_model_24khz()
    encoder.set_target_bandwidth(args.target_br)
    encoder.to(args.device)

    while True:
        data = encode_q.get()
        if data is None:
            if none_ctr == args.num_workers:
                ast_q.put(None)
                break
            continue
        data = deepcopy(data)
        out = encode_data(data["encodec_audio"], encoder, args)
        data["audio_tokens"] = out
        del data["encodec_audio"]
        ast_q.put(data)

    done.wait()


def ast_worker(ast_q, write_q, done, args):
    ast = ASTForAudioClassification.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    )
    ast.to(args.device)

    while True:
        data = ast_q.get()
        if data is None:
            write_q.put(None)
            break
        data = deepcopy(data)
        out = get_ast_logits(data["ast_audio"], ast, args)
        data["ast_logits"] = out
        del data["ast_audio"]
        write_q.put(data)

    done.wait()


def q_monitor(file_q, encode_q, ast_q, write_q, args, done):
    size = file_q.qsize()
    count = 0
    with tqdm(total=size, smoothing=0.01) as pbar:
        while True:
            if file_q.qsize() == 0 and encode_q.qsize() == 0 and write_q.qsize() == 0:
                break
            old = size
            size = file_q.qsize()
            pbar.update(old - size)
            if count % 10 == 0:
                tqdm.write(
                    f"File Queue: {file_q.qsize()} | Encode Queue: {encode_q.qsize():02d}/{args.batch_size} | AST Queue: {ast_q.qsize():02d}/{args.batch_size} | Write Queue: {write_q.qsize()}"
                )
            count += 1
            time.sleep(1)

    done.wait()


def write_worker(write_q, fp, done):
    while True:
        data = write_q.get()
        if data is None:
            break
        data = deepcopy(data)
        # write to disk
        for y, a, e, l in zip(
            data["ytid"], data["ast_logits"], data["audio_tokens"], data["labels"]
        ):
            out = {
                "ytid": y,
                "ast_logits": a.clone(),
                "audio_tokens": e.squeeze().clone(),
                "labels": l,
            }
            torch.save(out, os.path.join(fp, f"{y}.pt"))

    done.set()


def main(args):
    file_q = mp.Queue()
    encoder_q = mp.Queue(maxsize=args.batch_size)
    ast_q = mp.Queue(maxsize=args.batch_size)
    write_q = mp.Queue()
    done = mp.Event()

    meta = load_meta(args.meta)
    labels = load_labels(args.labels)

    workers = []
    processed = {
        os.path.basename(x).split(".")[0] for x in get_files(args.output_dir, ext=".pt")
    }

    for i, file in enumerate(get_files(args.input_dir)):
        ytid = os.path.basename(file).split(".")[0]
        if ytid in processed:
            continue
        file_q.put(file)

    workers.append(
        mp.Process(
            target=q_monitor,
            args=(file_q, encoder_q, ast_q, write_q, args, done),
        )
    )
    for i in range(args.num_workers):
        file_q.put(None)
        workers.append(
            mp.Process(
                target=load_worker,
                args=(file_q, meta, labels, encoder_q, args, done),
            )
        )
    workers.append(
        mp.Process(
            target=encode_worker,
            args=(encoder_q, ast_q, done, args),
        )
    )

    workers.append(mp.Process(target=ast_worker, args=(ast_q, write_q, done, args)))

    workers.append(
        mp.Process(target=write_worker, args=(write_q, args.output_dir, done))
    )

    for w in workers:
        w.start()

    done.wait()
    for w in workers:
        w.join()


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
        "--target_br", type=float, default=12.0, help="Target sample rate for encoder"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for inference"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="torch device to use",
    )
    parser.add_argument(
        "--target_device",
        type=int,
        default=0,
        help="Target GPU device to use. Only valid if device is cuda",
    )

    args = parser.parse_args()

    if args.device == "cuda":
        torch.cuda.set_device(args.target_device)

    torch.multiprocessing.set_start_method("spawn")
    torch.multiprocessing.set_sharing_strategy("file_system")

    main(args)
