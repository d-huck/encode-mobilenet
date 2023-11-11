import torch
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


class AudioSetDataset(Dataset):
    def __init__(self, fp: str, ext=".pt", device=None):
        super().__init__()
        self.files = []
        for root, dirs, files in os.walk(fp):
            for file in files:
                if file.endswith(ext):
                    self.files.append(os.path.join(root, file))

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

        # self.data, self.labels = [], []
        # for file in fp:
        #     data = torch.load(file)
        #     self.data.append(data["audio"])
        #     label = torch.tensor(data["labels"])
        #     self.labels = torch.sum(
        #         F.one_hot(label, num_classes=527), dim=0, dtype=torch.float32
        #     )

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

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
    valid_size = int(valid_size * len(data))
    train, valid = data[valid_size:], data[:valid_size]
    train = dataset_t(train, device=device)
    valid = dataset_t(valid, device=device)

    # x, y = data
    # x_train, x_test, y_train, y_test = train_test_split(
    #     x, y, test_size=test_size, random_state=random_seed
    # )
    # train = dataset_t(x_train, y_train, device=device)
    # test = dataset_t(x_test, y_test, device=device)

    train_loader = DataLoader(
        train, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    test_loader = DataLoader(
        valid, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    return train_loader, test_loader
