import torch
import numpy as np
from torch.utils.data import Dataset


class Custom_Dataset(Dataset):
    def __init__(self, data_dir, num_files):  # 样本地址，样本数量
        self.num_samples = num_files
        self.data_dir = data_dir

    def __getitem__(self, idx):
        file_idx = idx
        data_dict = torch.load(f"{self.data_dir}/{file_idx}.pth")
        return data_dict

    def __len__(self):
        return self.num_samples

