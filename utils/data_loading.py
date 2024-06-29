# -------------------------------------------------------------
# File: data_loading.py
# Author: Qiang Li
# Date of Completion: June 27, 2024
# Description: data loader
# -------------------------------------------------------------
# Input/Output Information (IO):
# Input: /
# Output: /
# -------------------------------------------------------------
import numpy as np
from torch.utils.data import Dataset
import torch
from matplotlib.path import Path
import os
from pathlib import Path


def count_subdirectories(base_dir):
    try:
        items = os.listdir(base_dir)
        subdirectories = [item for item in items if os.path.isdir(os.path.join(base_dir, item))]
        return len(subdirectories)
    except FileNotFoundError:
        print(f"Error: The folder '{base_dir}' does not exist.")
        return None


class MADDataset(Dataset):
    def __init__(self, data_dir: str, mask_dir: str):
        # reading simulation parameters
        self.data_dir = Path(data_dir)
        self.mask_dir = Path(mask_dir)
        files = os.listdir(data_dir)
        filtered_files = [file for file in files if file.startswith('case') and file.endswith('.npy')]
        self.ids = [str(file.replace('.npy', '')) for file in filtered_files]

    def __getitem__(self, index):
        name = self.ids[index]
        # mask_file = list(self.mask_dir.glob(name + '.*'))
        img_file = list(self.data_dir.glob(name + '.*'))
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        # assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        # mask = np.load(self.mask_dir.joinpath(f'{name}.npy'))
        data = np.load(self.data_dir.joinpath(f'{name}.npy'))
        # Bzz = np.load(self.data_dir.joinpath(f'Bzz_{name}.npy'))
        # assert mask.size * data.shape[0] == data.size, \
        #     f'data and mask in {name} should be the same size, but are {data.size} and {mask.size}'
        return {
            'image': torch.as_tensor(data.copy()).float().contiguous(),
            # 'mask': torch.as_tensor(mask.copy()).long().contiguous()
            # 'Bzz': torch.as_tensor(Bzz.copy()).float().contiguous()
        }

    def __len__(self):
        return len(self.ids)



