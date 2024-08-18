from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from functools import partial
import datetime as dt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer, Adam

from common.training import Accumulator, EarlyStopping, Timer, Logger, CheckpointSaver
from common.losses import RegularizedPowerError
from common.functional import compute_velocity_field
from common.plotting import plot_predictions_2d

from models.operators import GlobalOperator, LocalOperator
from era5.wind.datasets import Wind2dERA5



class GlobalOperatorPredictor:

    def __init__(
        self, 
        global_operator: GlobalOperator,
        device: torch.device,
    ):
        self.device: torch.device = device
        self.global_operator: GlobalOperator = global_operator.to(device=self.device)
        self.loss_function: nn.Module = nn.MSELoss(reduction='sum')

    def predict(self, dataset: Wind2dERA5, resolution: Tuple[int, int] | None) -> None:
        # Set the operator to evaluation mode
        self.global_operator.eval()
        # Batch size must be 1, must not shuffle
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        # Use only the first input from the test dataset:
        global_input: torch.Tensor = next(iter(dataloader))[0].to(device=self.device)
        # Compute how many bundle from last step to keep 
        n_retained_bundles: int = dataset.window_size - 1   # since each step predicts 1 bundle   

        # Keep track of groundtruths and predictions
        global_groundtruths: List[torch.Tensor] = []
        global_predictions: List[torch.Tensor] = []
        timestamps: List[str] = []
        metric_notes: List[str] = []

        with torch.no_grad():
            # Make multi-step prediction
            for bundle_id, (_, global_groundtruth) in enumerate(dataloader):
                # Make one-step prediction
                global_prediction: torch.Tensor = self.global_operator(input=global_input)[1]
                global_groundtruth: torch.Tensor = global_groundtruth.to(device=self.device)
                global_predictions.append(global_prediction)
                global_groundtruths.append(global_groundtruth)
                # Prepare input for next step
                if n_retained_bundles == 0:
                    global_input: torch.Tensor = global_prediction
                else: 
                    global_input: torch.Tensor = torch.cat(
                        tensors=[
                            global_input[:, -n_retained_bundles * dataset.bundle_size:, :, :, :],
                            global_prediction,
                        ],
                        dim=1
                    )
                assert global_prediction.shape == global_groundtruth.shape
                # Compute prediction timestamps
                prediction_timestamps: List[dt.datetime] = dataset.compute_timestamp(bundle_idx=bundle_id)[1]
                # Compute metrics
                total_mse: float = self.loss_function(input=global_prediction, target=global_groundtruth).item()
                mean_mse: float = total_mse / global_prediction.numel()
                mean_rmse: float = mean_mse ** 0.5
                timestamps.extend([f'{t.strftime("%Y-%m-%d %H:00")}' for t in prediction_timestamps])
                metric_notes.extend([f'MSE: {mean_mse:.4f}, RMSE: {mean_rmse:.4f}'] * dataset.bundle_size)

        global_predictions: torch.Tensor = torch.cat(tensors=global_predictions, dim=1).squeeze(dim=0)
        global_groundtruths: torch.Tensor = torch.cat(tensors=global_groundtruths, dim=1).squeeze(dim=0)
        assert global_predictions.shape[0] == global_groundtruths.shape[0] == len(metric_notes)

        # Plot the prediction
        plot_predictions_2d(
            groundtruths=global_groundtruths, 
            predictions=global_predictions, 
            timestamps=timestamps,
            metrics_notes=metric_notes, 
            reduction=partial(compute_velocity_field, dim=1),
            resolution=resolution
        )



if __name__ == '__main__':

    from common.training import CheckpointLoader
    global_loader = CheckpointLoader(checkpoint_path='.checkpoints/global/epoch245.pt')
    global_operator, _ = global_loader.load(scope=globals())

    self = GlobalOperatorPredictor(
        global_operator=global_operator,
        device=torch.device('cuda'),
    )

    dataset = Wind2dERA5(
        dataroot='data/2d/era5/wind',
        pressure_level=1000,
        fromdate='20240725',
        todate='20240731',
        global_latitude=(30, -10),
        global_longitude=(90, 130),
        global_resolution=(128, 128),
        local_latitude=None,
        local_longitude=None,
        local_resolution=None,
        bundle_size=12,
        window_size=1,
    )

    self.predict(dataset=dataset, resolution=(256, 256))

