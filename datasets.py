from typing import Tuple

import h5py
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class DiffReact2D(Dataset):

    def __init__(self, dataroot: str):
        self.dataroot: str = dataroot
        self.__file = h5py.File(name=dataroot, mode='r')
        self.__num_samples = len(self.__file.keys())

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        key: str = f'{str(idx).zfill(4)}/data'
        data: np.ndarray = np.array(self.__file[key], dtype=np.float16)
        data: torch.Tensor = torch.tensor(data).permute(0, 3, 1, 2)
        input_tensor: torch.Tensor = data[0]    # IC
        label_tensor: torch.Tensor = data[-1]   # last step
        return input_tensor, label_tensor

    def __len__(self) -> int:
        return self.__num_samples
    
    def __del__(self):
        self.__file.close()

if __name__ == '__main__':
    dataset = DiffReact2D(dataroot='data/2D/diffusion-reaction/2D_diff-react_NA_NA.h5')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    inputs, labels = next(iter(dataloader))





