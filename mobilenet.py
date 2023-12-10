from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from mobilenetv3.mobilenetv3 import hswish, hsigmoid, SeModule, Block
from encodec import EncodecModel
import torch
import torch.nn as nn
from torchmetrics.classification import MultilabelAveragePrecision
from torch.nn import init
import lightning as pl


class MobileNetV3_Smol(nn.Module):
    def __init__(self, encodec_bw=1.5, num_classes=10, act=nn.Hardswish):
        super(MobileNetV3_Smol, self).__init__()
        encoder = EncodecModel.encodec_model_24khz()
        encoder.set_target_bandwidth(encodec_bw)
        self.quantizer = encoder.quantizer
        self.quantizer.requires_grad = False

        # set up network
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


class MobileNetDistilled(pl.LightningModule):
    def __init__(self, encodec_bw=1.5, num_classes=10, a=1.0, b=1.0, act=nn.Hardswish):
        super().__init__()
        encoder = EncodecModel.encodec_model_24khz()
        encoder.set_target_bandwidth(encodec_bw)
        self.quantizer = encoder.quantizer
        self.quantizer.requires_grad = False
        self.map = MultilabelAveragePrecision(num_labels=num_classes)
        self.train_outputs = []
        self.valid_outputs = []
        self.test_outputs = []

        self.criterion = nn.BCEWithLogitsLoss()

        # set up network
        self.projection = nn.Sequential(
            nn.Conv2d(
                1,
                int(16 * a),
                kernel_size=(1, 3),
                stride=(1, 2),
                padding=(0, 0),
                bias=False,
            ),
            nn.BatchNorm2d(int(16 * a)),
            nn.ReLU(),
            nn.Conv2d(
                int(16 * a),
                int(16 * a),
                kernel_size=(1, 5),
                stride=(1, 3),
                padding=(0, 6),
                bias=False,
            ),
            nn.BatchNorm2d(int(16 * a)),
            nn.ReLU(),
        )

        self.conv1 = nn.Conv2d(
            int(16 * a), int(16 * a), kernel_size=21, stride=1, padding=2, bias=False
        )

        self.bn1 = nn.BatchNorm2d(int(16 * a))
        self.hs1 = act(inplace=True)

        self.bneck = nn.Sequential(
            Block(3, int(16 * a), int(16 * a), int(16 * a), nn.ReLU, False, 1),
            Block(3, int(16 * a), int(64 * a), int(24 * a), nn.ReLU, False, 2),
            Block(3, int(24 * a), int(72 * a), int(24 * a), nn.ReLU, False, 1),
            Block(5, int(24 * a), int(72 * a), int(40 * a), nn.ReLU, True, 2),
            Block(5, int(40 * a), int(120 * a), int(40 * a), nn.ReLU, True, 1),
            Block(5, int(40 * a), int(120 * a), int(40 * a), nn.ReLU, True, 1),
            Block(3, int(40 * a), int(240 * a), int(80 * a), act, False, 2),
            Block(3, int(80 * a), int(200 * a), int(80 * a), act, False, 1),
            Block(3, int(80 * a), int(184 * a), int(80 * a), act, False, 1),
            Block(3, int(80 * a), int(184 * a), int(80 * a), act, False, 1),
            Block(3, int(80 * a), int(480 * a), int(112 * a), act, True, 1),
            Block(3, int(112 * a), int(672 * a), int(112 * a), act, True, 1),
            Block(5, int(112 * a), int(672 * a), int(160 * a), act, True, 2),
            Block(5, int(160 * a), int(672 * a), int(160 * a), act, True, 1),
            Block(5, int(160 * a), int(960 * a), int(160 * a), act, True, 1),
        )

        self.conv2 = nn.Conv2d(
            int(160 * a), int(960 * a), kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(int(960 * a))
        self.hs2 = act(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.linear3 = nn.Linear(int(960 * a), int(1280 * a), bias=False)
        self.bn3 = nn.BatchNorm1d(int(1280 * a))
        self.hs3 = act(inplace=True)
        self.drop = nn.Dropout(0.2)

        self.linear4 = nn.Linear(int(1280 * a), num_classes)
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

    def configure_optimizers(self, args) -> OptimizerLRScheduler:
        optim = nn.BCEWithLogitsLoss(self.parameters)
        pct_start = args.warmup / args.epochs
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optim,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            pct_start=pct_start,
            max_lr=args.lr,
        )
        return self.optim

    def common_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = self.criterion(out, y)
        return {"loss": loss, "logits": out, "y": y}

    def training_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        outputs = self.common_step(batch, batch_idx)
        self.training_outputs.append(outputs)
        return outputs

    def on_training_epoch_end(self):
        outputs = self.train_outputs
        train_map = self.map(outputs["logits"], outputs["y"])
        epoch_loss = sum(outputs["loss"])
        batch_loss = epoch_loss / len(outputs["loss"])
        self.train_outputs = []
        self.log({"TL": epoch_loss, "aTL": batch_loss, "TmAP": train_map})

    def validation_step(
        self, batch, batch_idx, *args: Any, **kwargs: Any
    ) -> STEP_OUTPUT:
        outputs = self.common_step(batch, batch_idx)
        self.valid_outputs.append(outputs)
        return outputs

    def on_validation_epoch_end(self):
        outputs = self.valid_outputs
        valid_map = self.map(outputs["logits"], outputs["y"])
        epoch_loss = sum(outputs["loss"])
        batch_loss = epoch_loss / len(outputs["loss"])
        self.valid_outputs = []
        self.log({"VL": epoch_loss, "aVL": batch_loss, "TmAP": valid_map})

    def test_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        outputs = self.common_step(batch, batch_idx)
        self.test_outputs.append(outputs)
        return outputs

    def on_test_epoch_end(self, outputs) -> None:
        test_map = self.map(outputs["logits"], outputs["y"])
        self.log({"Testing mAP": test_map})


# first layer of mobilenet
class MobileNet(pl.LightningModule):
    def __init__(self, encodec_bw=1.5, num_classes=10, a=1.0, act=nn.Hardswish):
        super(MobileNet, self).__init__()
        encoder = EncodecModel.encodec_model_24khz()
        encoder.set_target_bandwidth(encodec_bw)
        self.quantizer = encoder.quantizer
        self.quantizer.requires_grad = False

        self.projection = nn.Sequential(
            nn.Conv2d(
                1,
                4,
                kernel_size=(1, 3),
                stride=(1, 2),
                padding=(0, 0),
                bias=False,
            ),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(
                4,
                16,
                kernel_size=(1, 5),
                stride=(1, 3),
                padding=(0, 6),
                bias=False,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.conv1 = nn.Conv2d(
            16, int(16 * a), kernel_size=3, stride=1, padding=2, bias=False
        )

        self.bn1 = nn.BatchNorm2d(int(16 * a))
        self.hs1 = act(inplace=True)

        self.bneck = nn.Sequential(
            Block(3, int(16 * a), int(16 * a), int(16 * a), nn.ReLU, False, 1),
            Block(3, int(16 * a), int(64 * a), int(24 * a), nn.ReLU, False, 2),
            Block(3, int(24 * a), int(72 * a), int(24 * a), nn.ReLU, False, 1),
            Block(5, int(24 * a), int(72 * a), int(40 * a), nn.ReLU, True, 2),
            Block(5, int(40 * a), int(120 * a), int(40 * a), nn.ReLU, True, 1),
            Block(5, int(40 * a), int(120 * a), int(40 * a), nn.ReLU, True, 1),
            Block(3, int(40 * a), int(240 * a), int(80 * a), act, False, 2),
            Block(3, int(80 * a), int(200 * a), int(80 * a), act, False, 1),
            Block(3, int(80 * a), int(184 * a), int(80 * a), act, False, 1),
            Block(3, int(80 * a), int(184 * a), int(80 * a), act, False, 1),
            Block(3, int(80 * a), int(480 * a), int(112 * a), act, True, 1),
            Block(3, int(112 * a), int(672 * a), int(112 * a), act, True, 1),
            Block(5, int(112 * a), int(672 * a), int(160 * a), act, True, 2),
            Block(5, int(160 * a), int(672 * a), int(160 * a), act, True, 1),
            Block(5, int(160 * a), int(960 * a), int(160 * a), act, True, 1),
        )

        self.conv2 = nn.Conv2d(
            int(160 * a), int(960 * a), kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(int(960 * a))
        self.hs2 = act(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.linear3 = nn.Linear(int(960 * a), 1280, bias=False)
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
        # print("before quantizer", x.shape)
        x = self.quantizer.decode(x)
        # print("after quantizer", x.shape)

        x = x.unsqueeze(1)  # add in a channel dimension
        # print("before projection", x.shape)
        x = self.projection(x)
        # print("after projection", x.shape)

        # run mobile net projection
        x = self.hs1(self.bn1(self.conv1(x)))

        # print("after second projection", x.shape)
        # run the bnet
        # x = self.bneck(x)
        all_features = torch.Tensor().cuda()
        for i, bneck_block in enumerate(self.bneck):
            block_num = i + 1

            x = bneck_block(x)
            # print(f"after block {block_num}", x.shape)

            if block_num in {5, 11, 13, 15}:
                features = x.detach()
                features = torch.mean(features, dim=3).view(features.shape[0], -1)
                # features = torch.mean(features, dim=(0, 3)).view(1, -1)
                all_features = torch.cat((all_features, features), dim=1)
        
        return all_features

        # # classify
        # x = self.hs2(self.bn2(self.conv2(x)))
        # x = self.gap(x).flatten(1)
        # x = self.drop(self.hs3(self.bn3(self.linear3(x))))

        # return self.linear4(x)
    
    @property
    def sample_rate(self):
        return 48000
