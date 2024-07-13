import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader


class DataCustom(Dataset):
    def __init__(
        self,
        file_path,
        dataset_name,
        dataset_name_wr,
        indices,
        mean,
        std,
        mean_wr,
        std_wr,
    ):
        self.file_path = file_path
        self.dataset_name = dataset_name
        self.dataset_name_wr = dataset_name_wr
        self.indices = indices
        self.mean = mean
        self.std = std
        self.mean_wr = mean_wr
        self.std_wr = std_wr

        self.file = h5py.File(self.file_path, "r")
        self.dataset = self.file[self.dataset_name]
        self.dataset_wr = self.file[self.dataset_name_wr]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        data_idx = self.indices[idx]
        data = self.dataset[data_idx]
        data_wr = self.dataset_wr[data_idx]

        data = (data - self.mean) / self.std
        data_wr = (data_wr - self.mean_wr) / self.std_wr

        return torch.tensor(data_wr, dtype=torch.float32).unsqueeze(0), torch.tensor(
            data, dtype=torch.float32
        ).unsqueeze(0)

    def close(self):
        self.file.close()

    # def __del__(self):
    #     self.close()


def get_dataloader(
    file_path,
    dataset_name,
    dataset_name_wr,
    indices,
    mean,
    std,
    mean_wr,
    std_wr,
    batch_size=32,
    shuffle=True,
    num_workers=4,
):
    dataset = DataCustom(
        file_path, dataset_name, dataset_name_wr, indices, mean, std, mean_wr, std_wr
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )


def get_shuffled_indices(n):
    indices = np.arange(n)
    np.random.shuffle(indices)
    return indices


def split_indices(indices, train_ratio=0.9):
    train_size = int(len(indices) * train_ratio)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    return train_indices, test_indices


def calculate_mean_std_online(file_path, dataset_name, dataset_name_wr, train_indices):
    with h5py.File(file_path, "r") as file:
        dataset = file[dataset_name]
        dataset_wr = file[dataset_name_wr]
        n = 0
        mean = 0.0
        M2 = 0.0
        mean_wr = 0.0
        M2_wr = 0.0

        for idx in train_indices:
            data = dataset[idx]
            data_wr = dataset_wr[idx]
            n += 1
            delta = data - mean
            delta_wr = data_wr - mean_wr
            mean += delta / n
            mean_wr += delta_wr / n
            delta2 = data - mean
            delta2_wr = data_wr - mean_wr
            M2 += delta * delta2
            M2_wr += delta_wr * delta2_wr

        variance = M2 / n
        variance_wr = M2_wr / n
        std_dev = np.sqrt(variance)
        std_dev_wr = np.sqrt(variance_wr)

    return np.mean(mean), np.mean(std_dev), np.mean(mean_wr), np.mean(std_dev_wr)


def data_provider(args):
    file_path = args.file_path

    # Get total number of data points
    # with h5py.File(file_path, 'r') as file:
    #     total_data_points = file[args.T21_wr_name].shape[0]

    total_data_points = 100000

    print("total data points", total_data_points)
    # Get shuffled and split indices
    indices = get_shuffled_indices(total_data_points)
    train_indices, test_indices = split_indices(indices, args.train_ratio)

    # Calculate mean and std for the training data
    # mean, std_dev, mean_wr, std_dev_wr = calculate_mean_std_online(file_path, args.T21_name, args.T21_wr_name, train_indices)
    mean, std_dev, mean_wr, std_dev_wr = (
        -14.241866,
        8.406500335592806,
        -2.7567148e-07,
        1.5758628076117043,
    )
    print("mean, std_dev, mean_wr, std_dev_wr", mean, std_dev, mean_wr, std_dev_wr)

    # Create DataLoaders for training and testing
    train_loader = get_dataloader(
        file_path,
        args.T21_name,
        args.T21_wr_name,
        train_indices,
        mean,
        std_dev,
        mean_wr,
        std_dev_wr,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = get_dataloader(
        file_path,
        args.T21_name,
        args.T21_wr_name,
        test_indices,
        mean,
        std_dev,
        mean_wr,
        std_dev_wr,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    return train_loader, test_loader
