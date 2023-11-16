import torch
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


def get_files(fp: str, ext=".flac") -> list:
    for root, dirr, files in os.walk(fp):
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
        try:
            data = torch.load(self.files[index])
        except:
            print("BAD FILE: ", self.files[index])
            return 0, 0  # cause an upstream error
        target = torch.tensor(data["labels"])
        target = torch.sum(
            F.one_hot(target, num_classes=527), dim=0, dtype=torch.float32
        )

        audio = data["audio_tokens"].squeeze()
        return audio, target


class GTZANDataset(Dataset):
    def __init__(self, data, labels, device=None):
        super().__init__()
        self.data = data

        self.labels = [torch.tensor(x) for x in labels]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        target = self.labels[index]

        return data, target


def split_data(
    data,
    batch_size=32,
    random_seed=42,
    device=None,
    valid_size=0.1,
    test_size=0.2,
    num_workers=8,
    dataset_t: Dataset = GTZANDataset,
    shuffle=True,
):
    # since the entire file list is shuffled before it goes through, this method
    # of splitting is still somewhat random.
    valid_size = int(valid_size * len(data))
    train, valid = data[valid_size:], data[:valid_size]
    train = dataset_t(train, device=device)
    valid = dataset_t(valid, device=device)

    train_loader = DataLoader(
        train, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    test_loader = DataLoader(
        valid, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    return train_loader, test_loader
