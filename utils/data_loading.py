import numpy as np
import logging
import pandas as pd
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
        mask_file = list(self.mask_dir.glob(name + '.*'))
        img_file = list(self.data_dir.glob(name + '.*'))
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = np.load(self.mask_dir.joinpath(f'{name}.npy'))
        data = np.load(self.data_dir.joinpath(f'{name}.npy'))
        assert mask.size * data.shape[0] == data.size, \
            f'data and mask in {name} should be the same size, but are {data.size} and {mask.size}'
        return {
            'image': torch.as_tensor(data.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
        }

    def __len__(self):
        return len(self.ids)




if __name__ == '__main__':
    dir_data = '../dataset_new/data/'
    dir_mask = '../dataset_new/mask/'
    # 将数据封装成 Dataset （用 TensorDataset 类）
    tensor_dataset = MADDataset(data_dir=dir_data, mask_dir=dir_mask)

    # 1. 测试数据集实例化和访问
    assert len(tensor_dataset) > 0, "数据集长度应该大于零"

    # 2. 测试数据加载
    sample_idx = 0
    sample = tensor_dataset[sample_idx]
    assert isinstance(sample['image'], torch.Tensor), "数据应该是一个张量"
    assert isinstance(sample['mask'], torch.Tensor), "目标应该是一个张量"

    # 3. 测试数据预处理（如果有）
    # ...

    # 4. 使用 DataLoader 进行批量加载
    # batch_size = 32
    # data_loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)
    # for batch_data, batch_targets in data_loader:
    #     assert batch_data.shape[0] == batch_size, "批次大小应该与指定的 batch_size 一致"
    #     assert batch_targets.shape[0] == batch_size, "批次大小应该与指定的 batch_size 一致"
    #     break  # 只检查一个批次

