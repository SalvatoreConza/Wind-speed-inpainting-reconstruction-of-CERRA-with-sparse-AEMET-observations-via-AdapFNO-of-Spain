from typing import Tuple, List
import datetime as dt

from functools import lru_cache
import xarray as xr

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class ERA5_6Hour(Dataset):

    def __init__(
        self,
        dataroot: str,
        fromdate: str,
        todate: str,
        global_latitude: Tuple[float, float] | None,
        global_longitude: Tuple[float, float] | None,
        global_resolution: Tuple[int, int] | None,
        local_latitude: Tuple[float, float] | None,
        local_longitude: Tuple[float, float] | None,
        indays: int,
        outdays: int,
    ):
        """
        latitude: (a, b) in the range [90.0, 89.75, 89.5, ..., -89.5, -89.75, -90.0]
        longitude: (a, b) in the range [0.0, 0.25, 0.5, ..., 359.25, 359.5, 359.75]
        """
        super().__init__()
        self.dataroot: str = dataroot
        self.fromdate: dt.datetime = dt.datetime.strptime(fromdate, '%Y%m%d')
        self.todate: dt.datetime = dt.datetime.strptime(todate, '%Y%m%d')
        self.global_latitude: Tuple[int, int] | None = global_latitude
        self.global_longitude: Tuple[int, int] | None = global_longitude
        self.local_latitude: Tuple[int, int] | None = local_latitude
        self.local_longitude: Tuple[int, int] | None = local_longitude
        self.indays: int = indays
        self.outdays: int = outdays
        
        self.in_channels: int = 20
        self.out_channels: int = 2

        self.time_resolution: int = 6
        self.timesteps_per_day: int = 24 // self.time_resolution
        self.in_timesteps: int = self.timesteps_per_day * indays
        self.out_timesteps: int = self.timesteps_per_day * outdays
        
        self.dataset: xr.Dataset = xr.open_dataset(dataroot, engine='cfgrib').sortby('time')
        self.dataset = self.dataset.sel(
            time=slice(self.fromdate, self.todate + dt.timedelta(days=1) - dt.timedelta(seconds=1))
        )

        self.total_timesteps: int = len(self.dataset.time)
        self.total_days: int = self.total_timesteps // self.timesteps_per_day

        self.slices: List[Tuple[slice, slice]] = [
            (slice(s, s + self.in_timesteps), slice(s + self.in_timesteps, s + self.in_timesteps + self.out_timesteps))
            for s in range(0, self.total_timesteps - self.in_timesteps + 1, self.out_timesteps)
                if s + self.in_timesteps + self.out_timesteps <= self.total_timesteps   # drop incomplete slice
        ]
        assert len(self.slices) == len(self)

        # Check global/local dataset
        self.has_global: bool = all([global_latitude, global_longitude])
        self.has_local: bool = all([local_latitude, local_longitude])
        assert self.has_global or self.has_local, 'either global or local must be specified'

        # Precompute sub-dataset for faster indexing (xr.Dataset does lazy loading on )
        if self.has_global:
            global_dataset: xr.Dataset = self.dataset.sel(
                latitude=slice(*self.global_latitude), longitude=slice(*self.global_longitude),
            )
            # Prepare global_subsets which contains (global input subset, global output subset)
            global_subsets: List[Tuple[xr.Dataset, xr.Dataset]] = [
                (global_dataset.isel(time=input_slice), global_dataset.isel(time=output_slice))
                for input_slice, output_slice in self.slices
            ]
            # Prepare global_subsets which contains (surface subset, pressure subset, output subset)
            self.global_subsets: List[Tuple[xr.Dataset, xr.Dataset, xr.Dataset]] = [
                (
                    global_input[['u10', 'v10', 't2m', 'msl', 'sp']], 
                    global_input.sel(isobaricInhPa=[1000, 850, 500])[['z', 'r', 't', 'u', 'v']], 
                    global_output[['u10', 'v10']],
                )
                for global_input, global_output in global_subsets
            ]

        if self.has_local:
            local_dataset: xr.Dataset = self.dataset.sel(
                latitude=slice(*self.local_latitude), longitude=slice(*self.local_longitude),
            )
            # Prepare local_subsets which contains (local input subset, local output subset)
            local_subsets: List[Tuple[xr.Dataset, xr.Dataset]] = [
                (local_dataset.isel(time=input_slice), local_dataset.isel(time=output_slice))
                for input_slice, output_slice in self.slices
            ]
            # Prepare local_subsets which contains (surface subset, pressure subset, output subset)
            self.local_subsets: List[Tuple[xr.Dataset, xr.Dataset, xr.Dataset]] = [
                (
                    local_input[['u10', 'v10', 't2m', 'msl', 'sp']], 
                    local_input.sel(isobaricInhPa=[1000, 850, 500])[['z', 'r', 't', 'u', 'v']], 
                    local_output[['u10', 'v10']],
                )
                for local_input, local_output in local_subsets
            ]

        # Compute resolution
        if self.has_global:
            if global_resolution is None:
                self.global_resolution: Tuple[int, int] = (
                    (self.global_latitude[0] - self.global_latitude[1]) * 4 + 1,
                    (self.global_longitude[1] - self.global_longitude[0]) * 4 + 1
                )
            else:
                self.global_resolution: Tuple[int, int] = global_resolution
        else:
            self.global_resolution = None

        if self.has_local:
            self.local_resolution: Tuple[int, int] = (
                    (self.local_latitude[0] - self.local_latitude[1]) * 4 + 1,
                    (self.local_longitude[1] - self.local_longitude[0]) * 4 + 1
                )
        else:
            self.local_resolution = None

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        sample: Tuple[torch.Tensor, ...] = tuple()

        if self.has_global:
            global_input, global_output = self._to_tensor(idx=idx, is_global=True)
            sample += (global_input, global_output)

        if self.has_local:
            local_input, local_output = self._to_tensor(idx=idx, is_global=False)
            sample += (local_input, local_output)
        
        return sample

    def __len__(self) -> int:
        return (self.total_days - self.indays) // self.outdays

    @lru_cache(maxsize=10240)
    def _to_tensor(self, idx: int, is_global: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        if is_global:
            surface_subset, pressure_subset, output_subset = self.global_subsets[idx]
            resize: Tuple[int, int] = self.global_resolution
        else:
            surface_subset, pressure_subset, output_subset = self.local_subsets[idx]
            resize = None

        # Process surface tensor
        surface_level_tensor: torch.Tensor = torch.tensor(surface_subset.to_dataarray().values)
        surface_level_tensor = surface_level_tensor.permute(1, 0, 2, 3)
        # Process pressure tensor
        pressure_level_tensor: torch.Tensor = torch.tensor(pressure_subset.to_dataarray().values)
        pressure_level_tensor = pressure_level_tensor.permute(1, 2, 0, 3, 4).flatten(start_dim=1, end_dim=2)
        # Merge to input tensor
        in_tensor: torch.Tensor = torch.cat(tensors=[surface_level_tensor, pressure_level_tensor], dim=1)
        assert in_tensor.ndim == 4 and in_tensor.shape == (
            self.in_timesteps, self.in_channels, in_tensor.shape[2], in_tensor.shape[3]
        )
        # Resize
        if (resize is not None) and (resize != tuple(in_tensor.shape[-2:])):
            in_tensor = F.interpolate(input=in_tensor, size=resize, mode='bicubic')
        
        # Process output tensor
        out_tensor: torch.Tensor = torch.tensor(output_subset.to_dataarray().values)
        out_tensor = out_tensor.permute(1, 0, 2, 3)
        assert out_tensor.ndim == 4 and out_tensor.shape == (
            self.out_timesteps, self.out_channels, out_tensor.shape[2], out_tensor.shape[3]
        )
        # Resize
        if (resize is not None) and (resize != tuple(out_tensor.shape[-2:])):
            out_tensor = F.interpolate(input=out_tensor, size=resize, mode='bicubic')

        return in_tensor, out_tensor


if __name__ == '__main__':

    self = ERA5_6Hour(
        dataroot='./data/out.grib',
        fromdate='20240101',
        todate='20240630',
        global_latitude=(5, -5),
        global_longitude=(80, 90),
        global_resolution=(41, 41),
        local_latitude=(2, -2),
        local_longitude=(86, 90),
        indays=3,
        outdays=2,
    )

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset=self, batch_size=8, num_workers=4)
    for sample in dataloader:
        global_input, global_output, local_input, local_output = sample
        print(global_input.shape)
        print(global_output.shape)
        print(local_input.shape)
        print(local_output.shape)
        print('-----')






