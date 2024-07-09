import sys
from typing import List, Dict
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from neuralop import Trainer
from neuralop import LpLoss, H1Loss
from neuralop.models import FNO, FNO2d

from datasets import Burgers2D


class FNO2DTrainer:

    def __init__(
        self, 
        model: FNO,
        train_dataset: Dataset,
        test_dataset: Dataset,
        optimizer: torch.optim.Optimizer,
        n_epochs: int,
    ) -> None:
        self.model: FNO = model
        self.spatial_dim: int = model.out_channels
        self.train_dataset: Dataset = train_dataset
        self.test_dataset: Dataset = test_dataset
        self.train_loader: DataLoader = DataLoader(dataset=train_dataset, batch_size=100)
        self.test_loader: DataLoader = DataLoader(dataset=test_dataset, batch_size=100)
        self.optimizer: Optimizer = optimizer
        self.scheduler: CosineAnnealingLR = CosineAnnealingLR(optimizer, T_max=30)
        self.h1loss: H1Loss = H1Loss(d=self.spatial_dim, reduce_dims=[0, 1])
        self.l2loss: LpLoss = LpLoss(d=self.spatial_dim, p=2)
        self.n_epochs: int = n_epochs
        self.trainer: Trainer = Trainer(
            model=model, 
            n_epochs=n_epochs,
            wandb_log=False,
            log_test_interval=3,
            use_distributed=False,
            verbose=True,
        )

    def train(self) -> None:
        self.trainer.train(
            train_loader=self.train_loader,
            test_loaders={64: self.test_loader},
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            regularizer=False,
            training_loss=self.h1loss,
            eval_losses={'h1': self.h1loss, 'l2': self.l2loss},
        )

    def plot_eval(self, at_steps: List[int]):
        n_steps: int = len(at_steps)

        test_samples: Dict[str, torch.Tesnor] = next(iter(self.test_loader))
        input_tensor: torch.Tensor = test_samples['x']
        gt_tensor: torch.Tensor = test_samples['y']
        pred_tensor: torch.Tensor = self.model(x=input_tensor)

        fig = plt.figure(figsize=(11, 3.5 * n_steps + 1))
        for i in range(n_steps):
            time_index: int = at_steps[i] - 1

            input_sample: torch.Tensor = input_tensor[time_index]
            input_sample: np.ndarray = input_sample.permute(1, 2, 0).detach().numpy()
            input_u: np.ndarray = (input_sample ** 2).sum(axis=2) ** 0.5

            gt_sample: torch.Tensor = gt_tensor[time_index]
            gt_sample: np.ndarray = gt_sample.permute(1, 2, 0).detach().numpy()
            gt_u: np.ndarray = (gt_sample ** 2).sum(axis=2) ** 0.5
            
            pred_sample: torch.Tensor = pred_tensor[time_index]
            pred_sample: np.ndarray = pred_sample.permute(1, 2, 0).detach().numpy()
            pred_u: np.ndarray = (pred_sample ** 2).sum(axis=2) ** 0.5
            
            ax = fig.add_subplot(n_steps, 3, i * 3 + 1)
            ax.imshow(input_u)
            ax.set_title(f'input velocity field t=0')

            ax = fig.add_subplot(n_steps, 3, i * 3 + 2)
            ax.imshow(gt_u)
            ax.set_title(f'ground-truth velocity field t={time_index + 1}')

            ax = fig.add_subplot(n_steps, 3, i * 3 + 3)
            ax.imshow(pred_u)
            ax.set_title(f'predicted velocity field t={time_index + 1}')

        plt.tight_layout()
        t = (
            f'R={train_dataset.R}, ' 
            f'a={train_dataset.a}, '
            f'w={train_dataset.w}, '
            f'dropout_rate={self.train_dataset.dropout_rate}, '
            f'epochs={self.n_epochs}'
        )
        plt.suptitle(t, x=0.5, y=1.0, va='top')
        plt.savefig(
            f'{train_dataset.R}_'
            f'{train_dataset.a}_'
            f'{train_dataset.w}_'
            f'{self.train_dataset.dropout_rate}_'
            f'{self.n_epochs}.png'
        )
        plt.close()
        return pred_tensor

if __name__ == '__main__':

    train_dataset = Burgers2D(is_train=True, R=2500, a=0.8, w=1., dropout_rate=0.3)
    test_dataset = Burgers2D(is_train=True, R=2500, a=0.8, w=1., dropout_rate=0.)

    model = FNO2d(
        n_modes_height=16, 
        n_modes_width=16, 
        hidden_channels=16, 
        in_channels=2, 
        out_channels=2, 
    )
    optimizer = Adam(
        model.parameters(),
        lr=8e-3,
        weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=30)

    trainer = FNO2DTrainer(
        model=model, 
        train_dataset=train_dataset, 
        test_dataset=test_dataset, 
        optimizer=optimizer, 
        n_epochs=50, 
    )
    trainer.train()

    a = trainer.plot_eval(at_steps=[100])



from neuralop.datasets import load_darcy_flow_small