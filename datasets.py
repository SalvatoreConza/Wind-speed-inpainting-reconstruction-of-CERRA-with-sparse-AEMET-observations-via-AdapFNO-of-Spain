import os
from typing import List
import random

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class Burgers2D(Dataset):

    def __init__(self, is_train: bool, R: float, a: float, w: float, dropout_rate: float) -> None:
        super().__init__()
        self.is_train: bool = is_train
        self.R: float = R
        self.a: float = a
        self.w: float = w
        self.dropout_rate: float = dropout_rate

        if is_train:
            data_path: str = './data/burgers/train/'
            prefix: str = 'burgers_train_'
        else:
            data_path: str = './data/burgers/test/'
            prefix: str = 'burgers_test_'

        suffix: str = f'{int(R)}_{int(a * 10)}_{int(w * 10)}.npy'
        
        filename: str = prefix + suffix

        data_tensor: torch.Tensor = torch.tensor(
            data=np.load(file=data_path + filename), 
            dtype=torch.float,
        )
        data_tensor: torch.Tensor = data_tensor.permute(2, 3, 0, 1) # (timesteps, dim, x, y)
        prob_tensor: torch.Tensor = torch.full(data_tensor.shape, fill_value=(1 - dropout_rate))
        mask: torch.Tensor = torch.bernoulli(input=prob_tensor)
        data_tensor: torch.Tensor = data_tensor * mask

        self.states: torch.Tensor = data_tensor[1:].clone()
        timesteps: int = self.states.shape[0]
        
        self.ic: torch.Tensor = data_tensor[:1].clone()
        self.ic: torch.Tensor = self.ic.repeat(timesteps, 1, 1, 1)
        assert self.ic.shape == self.states.shape

    def __getitem__(self, idx: int):
        return {'x': self.ic[idx], 'y': self.states[idx]}
    
    def __len__(self):
        return self.ic.shape[0]
    


if __name__ == '__main__':
    train_dataset = Burgers2D(is_train=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=32)



