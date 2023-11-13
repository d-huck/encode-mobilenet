import torch
import os
import torchaudio
from torchaudio.transforms import Resample
from hear21passt.base import load_model
from dataset import get_files

import multiprocessing as mp


def load_worker(file_q, logits_q, pt_dir, target_sr):
    resampler = Resample(
        orig_freq=48000,
        new_freq=16000,
        resampling_method="sinc_kaiser_best",
        lowpass_filter_width=64,
        rolloff=0.9475937167399596,
        beta=14.769656459379492,
    )
    while True:
        fp = file_q.get()
        if fp is None:
            logits_q.put(None)
            break
        ytid = os.path.basename(fp).split(".")[0]
        audio, sr = torchaudio.load(fp)
        audio = resampler(audio)

def passt_worker(logits_q, write_q):
    pass

if __name__ == "__main__":
    file_q = mp.Queue()
    logits_q = mp.Queue()
    test_file = "/storage/datasets/"
    file_q.put("/storage/datasets/Au")