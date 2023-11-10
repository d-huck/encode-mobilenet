import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class AudioSetDataset(Dataset):
    def __init__(self, data, device=None):
        super().__init__()
        self.data = data["data"]
        labels = data["targets"]
        self.labels = [torch.tensor(x) for x in labels]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        target = self.labels[index]

        target = torch.sum(
            torch.nn.functional.one_hot(target, num_classes=527),
            dim=0,
            dtype=torch.float32,
        )
        return data, target


class AudioSetEpoch(Dataset):
    def __init__(self, data, labels, device=None):
        super().__init__()
        self.data = data
        self.labels = labels

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index].to(self.device)
        target = self.labels[index].to(self.device)

        return data, target


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
    dataset_t: Dataset = GTZANDataset,
    shuffle=True,
):
    x, y = data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_seed
    )

    train = dataset_t(x_train, y_train, device=device)
    test = dataset_t(x_test, y_test, device=device)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader
