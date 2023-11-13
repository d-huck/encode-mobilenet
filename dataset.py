from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import lightning as pl


def get_files(fp: str, ext=".flac") -> list:
    for root, dir, files in os.walk(fp):
        for f in files:
            if f.endswith(ext):
                yield os.path.join(root, f)


class AudioSetDataset(Dataset):
    """Entire Audioset Dataset. Expects a directory of saved torch tensors that can be
    loaded at will and served to mobilenet.
    """

    def __init__(self, fp: str, ext=".pt", device=None):
        super().__init__()
        self.files = [file for file in get_files(fp, ext=ext)]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return self.files[index]


class AudioSetEpoch(Dataset):
    """Expects a list of filepointers of saved torch tensors that can be loaded at
    will and served to mobilenet.

    :param Dataset: _description_
    :type Dataset: _type_
    """

    def __init__(self, fp: list, device=None):
        super().__init__()
        self.files = fp

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data = torch.load(self.files[index])
        target = torch.tensor(data["labels"])
        target = torch.sum(
            F.one_hot(target, num_classes=527), dim=0, dtype=torch.float32
        )

        audio = data["audio"].squeeze()
        return audio, target


class AudioSetDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers=4):
        super().__init__()
        if self.data_dir.endswith("/"):
            self.data_dir = data_dir[:-1]
        else:
            self.data_dir = data_dir

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ds = None
        self.valid_ds = None
        self.test_ds = None

    def setup(self, stage):
        train_data = get_files(f"{self.data_dir}/unbalanced", ext=".pt")
        valid_data = get_files(f"{self.data_dir}/balanced", ext=".pt")
        test_data = get_files(f"{self.data_dir}/eval", ext=".pt")
        self.train_ds = AudioSetEpoch(train_data)
        self.valid_ds = AudioSetEpoch(valid_data)
        self.test_data = AudioSetEpoch(test_data)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_ds, self.batch_size, shuffle=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.valid_ds, self.batch_size, shuffle=False)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.train_ds, self.batch_size, shuffle=False)
