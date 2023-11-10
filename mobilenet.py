from mobilenetv3.mobilenetv3 import hswish, hsigmoid, SeModule, Block
from encodec import EncodecModel
import torch
import torch.nn as nn
from torch.nn import init


class MobileNetV3_Smol(nn.Module):
    def __init__(self, encodec_bw=1.5, num_classes=10, act=nn.Hardswish):
        super(MobileNetV3_Smol, self).__init__()
        encoder = EncodecModel.encodec_model_24khz()
        encoder.set_target_bandwidth(encodec_bw)
        self.quantizer = encoder.quantizer
        self.quantizer.requires_grad = False

        self.projection = nn.Sequential(
            nn.ConvTranspose2d(
                1, 3, kernel_size=(2, 3), stride=(2, 1), padding=(16, 264), bias=False
            ),
            nn.BatchNorm2d(3),
            act(inplace=True),
        )
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = act(inplace=True)

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU, True, 2),
            Block(3, 16, 72, 24, nn.ReLU, False, 2),
            Block(3, 24, 88, 24, nn.ReLU, False, 1),
            Block(5, 24, 96, 40, act, True, 2),
            Block(5, 40, 240, 40, act, True, 1),
            Block(5, 40, 240, 40, act, True, 1),
            Block(5, 40, 120, 48, act, True, 1),
            Block(5, 48, 144, 48, act, True, 1),
            Block(5, 48, 288, 96, act, True, 2),
            Block(5, 96, 576, 96, act, True, 1),
            Block(5, 96, 576, 96, act, True, 1),
        )

        self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(576)
        self.hs2 = act(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.linear3 = nn.Linear(576, 1280, bias=False)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = act(inplace=True)
        self.drop = nn.Dropout(0.2)
        self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # decode from the encodec representation
        x = x.transpose(0, 1)
        x = self.quantizer.decode(x)

        x = x.unsqueeze(1)  # add in a channel dimension
        x = self.projection(x)

        # run mobile net projection
        x = self.hs1(self.bn1(self.conv1(x)))

        # run the bnet
        x = self.bneck(x)

        # classify
        x = self.hs2(self.bn2(self.conv2(x)))
        x = self.gap(x).flatten(1)
        x = self.drop(self.hs3(self.bn3(self.linear3(x))))

        return self.linear4(x)


# first layer of mobilenet
class MobileNetV3_LARGE(nn.Module):
    def __init__(self, encodec_bw=1.5, num_classes=10, act=nn.Hardswish):
        super(MobileNetV3_LARGE, self).__init__()
        encoder = EncodecModel.encodec_model_24khz()
        encoder.set_target_bandwidth(encodec_bw)
        self.quantizer = encoder.quantizer
        self.quantizer.requires_grad = False

        self.projection = nn.Sequential(
            nn.Conv2d(
                1, 16, kernel_size=(1, 3), stride=(1, 2), padding=(0, 0), bias=False
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(
                16, 16, kernel_size=(1, 5), stride=(1, 3), padding=(0, 6), bias=False
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.conv1 = nn.Conv2d(16, 16, kernel_size=21, stride=1, padding=2, bias=False)

        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = act(inplace=True)

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU, False, 1),
            Block(3, 16, 64, 24, nn.ReLU, False, 2),
            Block(3, 24, 72, 24, nn.ReLU, False, 1),
            Block(5, 24, 72, 40, nn.ReLU, True, 2),
            Block(5, 40, 120, 40, nn.ReLU, True, 1),
            Block(5, 40, 120, 40, nn.ReLU, True, 1),
            Block(3, 40, 240, 80, act, False, 2),
            Block(3, 80, 200, 80, act, False, 1),
            Block(3, 80, 184, 80, act, False, 1),
            Block(3, 80, 184, 80, act, False, 1),
            Block(3, 80, 480, 112, act, True, 1),
            Block(3, 112, 672, 112, act, True, 1),
            Block(5, 112, 672, 160, act, True, 2),
            Block(5, 160, 672, 160, act, True, 1),
            Block(5, 160, 960, 160, act, True, 1),
        )

        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = act(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.linear3 = nn.Linear(960, 1280, bias=False)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = act(inplace=True)
        self.drop = nn.Dropout(0.2)

        self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # decode from the encodec representation
        x = x.transpose(0, 1)
        x = self.quantizer.decode(x)

        x = x.unsqueeze(1)  # add in a channel dimension
        x = self.projection(x)

        # run mobile net projection
        x = self.hs1(self.bn1(self.conv1(x)))

        # run the bnet
        x = self.bneck(x)

        # classify
        x = self.hs2(self.bn2(self.conv2(x)))
        x = self.gap(x).flatten(1)
        x = self.drop(self.hs3(self.bn3(self.linear3(x))))

        return self.linear4(x)
