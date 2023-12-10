import torch

import sys
import os
sys.path.append(os.path.dirname("../"))
from mobilenet import MobileNet
from preprocess import encode_data

from encodec import EncodecModel
from encodec.utils import convert_audio

def load_model():
    model = MobileNet(encodec_bw=12, num_classes=527, a=4.0)
    model.load_state_dict(torch.load("/saltpool0/data/davin/encodec-model-a4.0-kd0.1-best.pt"))
    return model

encoder = EncodecModel.encodec_model_24khz()
encoder.set_target_bandwidth(12.0)

def get_scene_embeddings(audio, model):
    audio = audio.unsqueeze(1)
    audio = convert_audio(wav=audio.cpu(), sr=48000, target_sr=24000, target_channels=1)

    codes = encode_data(audio, encoder.cuda(), "cuda:0")

    embeddings = model(codes.cuda())
    return embeddings
