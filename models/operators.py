from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import (
    AdaptiveSpectralConv2d, ContextAggregateLayer, FeatureNormalization, 
    LiftingLayer, LocalLinearTransformation, TemporalAggregateLayer, TemporalAttention
)

class _BaseOperator(nn.Module):

    def __init__(
        self, 
        bundle_size: int, 
        window_size: int, 
        u_dim: int, 
        width: int, depth: int, 
        x_modes: int, y_modes: int,
    ):
        super().__init__()

        self.bundle_size: int = bundle_size
        self.window_size: int = window_size
        self.in_timesteps: int = bundle_size * window_size
        self.out_timesteps: int = bundle_size
        self.u_dim: int = u_dim
        self.width: int = width
        self.depth: int = depth
        self.x_modes: int = x_modes
        self.y_modes: int = y_modes
        
        assert width > u_dim, '`width` should be greater than `u_dim` for the model to uplift the input dim'
        self.P = LiftingLayer(in_features=self.u_dim, out_features=self.width)
        self.Q = LiftingLayer(in_features=self.width, out_features=self.u_dim)
        self.Wt = TemporalAggregateLayer(in_timesteps=self.in_timesteps, out_timesteps=self.out_timesteps)
        # self.Wt = TemporalAttention(
        #     in_timesteps=self.in_timesteps, out_timesteps=self.out_timesteps, 
        #     width=self.width, x_res=128, y_res=128,
        #     downsampling_factor=16,
        #     n_heads=8, dropout=0.,
        # )

        self.spectral_convolutions = nn.ModuleList(modules=[])
        self.local_linear_transformations = nn.ModuleList(modules=[])
        self.feature_normalizations = nn.ModuleList(modules=[])
        
        for _ in range(depth):
            self.spectral_convolutions.append(
                AdaptiveSpectralConv2d(u_dim=width, x_modes=x_modes, y_modes=y_modes)
            )
            self.local_linear_transformations.append(LocalLinearTransformation(u_dim=width))
            self.feature_normalizations.append(FeatureNormalization(normalized_shape=(self.width,), dims=(2,)))


class GlobalOperator(_BaseOperator):

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # input dim = [batch_size, in_timesteps, u_dim, x_res, y_res)
        batch_size: int = input.shape[0]
        x_res: int = input.shape[3]
        y_res: int = input.shape[4]
        assert self.in_timesteps == input.shape[1]
        assert self.u_dim == input.shape[2]

        # UpLifting
        lifted_input: torch.Tensor = self.P(input)
        assert lifted_input.shape == (batch_size, self.in_timesteps, self.width, x_res, y_res)

        # Fourier Layers
        fourier_outputs: List[torch.Tensor] = [lifted_input]
        for i in range(self.depth):
            fourier_output: torch.Tensor = fourier_outputs[-1]
            # Apply spectral convolution
            spectral_conv: AdaptiveSpectralConv2d = self.spectral_convolutions[i]
            out1: torch.Tensor = spectral_conv(fourier_output)
            # Apply local linear transformation
            local_linear_tranformation: LocalLinearTransformation = self.local_linear_transformations[i]
            out2: torch.Tensor = local_linear_tranformation(fourier_output)
            # Connection
            assert out1.shape == out2.shape == (batch_size, self.in_timesteps, self.width, x_res, y_res), (
                f'both out1 and out2 must have the same shape as '
                f'(batch_size, self.in_timesteps, self.width, self.x_res, self.y_res), '
                f'got out1.shape = {out1.shape} and out2.shape = {out2.shape}'
            )
            fourier_output = out1 + out2

            # Normalize over the width axis
            feature_normalization: FeatureNormalization = self.feature_normalizations[i]
            fourier_output = feature_normalization(fourier_output)
            # Apply non-linearity
            fourier_output = F.gelu(fourier_output)
            assert fourier_output.shape == (batch_size, self.in_timesteps, self.width, x_res, y_res)
            # Append fourier outputs
            fourier_outputs.append(fourier_output)

        global_contexts: List[torch.Tensor] = fourier_outputs[1:] # skip lifted_input

        # Apply temporal weights
        weighted_fourier_output: torch.Tensor = self.Wt(fourier_output) # applied on last fourier output
        assert weighted_fourier_output.shape == (batch_size, self.out_timesteps, self.width, x_res, y_res)
        # Projection
        projected_output: torch.Tensor = self.Q(weighted_fourier_output)
        assert projected_output.shape == (batch_size, self.out_timesteps, self.u_dim, x_res, y_res)
        return projected_output, *global_contexts


class LocalOperator(_BaseOperator):

    def __init__(
        self, 
        bundle_size: int, 
        window_size: int, 
        u_dim: int, 
        width: int, depth: int, 
        x_modes: int, y_modes: int,
        x_res: int, y_res: int,
    ):
        super().__init__(
            bundle_size=bundle_size, window_size=window_size, 
            u_dim=u_dim, x_modes=x_modes, y_modes=y_modes,
            width=width, depth=depth, 
        )
        # NOTE: LocalOperator.width == GlobalOperator.width
        self.x_res: int = x_res
        self.y_res: int = y_res
        self.context_aggregate_layers = nn.ModuleList([
            ContextAggregateLayer(local_x_res=self.x_res, local_y_res=self.y_res)
            for _ in range(self.depth)
        ])

    def forward(
        self, 
        input: torch.Tensor, 
        global_contexts: List[torch.Tensor],
    ) -> torch.Tensor:
        
        # input dim = [batch_size, in_timesteps, u_dim, x_res, y_res)
        batch_size: int = input.shape[0]
        assert input.shape == (batch_size, self.in_timesteps, self.u_dim, self.x_res, self.y_res)
        assert self.width == global_contexts[0].shape[2], 'LocalOperator.width must be equal to GlobalOperator.width'

        # UpLifting
        lifted_input: torch.Tensor = self.P(input)
        assert lifted_input.shape == (batch_size, self.in_timesteps, self.width, self.x_res, self.y_res)

        # Fourier Layers
        fourier_output: torch.Tensor = lifted_input
        for i in range(self.depth):
            # Apply spectral convolution
            spectral_conv: AdaptiveSpectralConv2d = self.spectral_convolutions[i]
            out1: torch.Tensor = spectral_conv(fourier_output)
            # Apply local linear transformation
            local_linear_tranformation: LocalLinearTransformation = self.local_linear_transformations[i]
            out2: torch.Tensor = local_linear_tranformation(fourier_output)
            # Connection
            assert out1.shape == out2.shape == (batch_size, self.in_timesteps, self.width, self.x_res, self.y_res), (
                f'both out1 and out2 must have the same shape as '
                f'(batch_size, self.in_timesteps, self.width, self.x_res, self.y_res), '
                f'got out1.shape = {out1.shape} and out2.shape = {out2.shape}'
            )
            fourier_output = out1 + out2

            # Normalize over temporal and width axes
            feature_normalization: FeatureNormalization = self.feature_normalizations[i]
            fourier_output = feature_normalization(fourier_output)
            # Apply non-linearity
            fourier_output = F.gelu(fourier_output)

            # Condition on global context
            context_aggregate_layer: ContextAggregateLayer = self.context_aggregate_layers[i]
            fourier_output = context_aggregate_layer(global_context=global_contexts[i], local_context=fourier_output)

        assert fourier_output.shape == (batch_size, self.in_timesteps, self.width, self.x_res, self.y_res)

        # Apply temporal weights
        weighted_fourier_output: torch.Tensor = self.Wt(fourier_output)
        assert weighted_fourier_output.shape == (batch_size, self.out_timesteps, self.width, self.x_res, self.y_res)
        # Projection
        projected_output: torch.Tensor = self.Q(weighted_fourier_output)
        assert projected_output.shape == (batch_size, self.out_timesteps, self.u_dim, self.x_res, self.y_res)
        return projected_output


if __name__ == '__main__':

    device = torch.device('cuda')
    global_input: torch.Tensor = torch.rand((32, 12, 2, 96, 96)).to(device)
    local_input: torch.Tensor = torch.rand((32, 12, 2, 64, 64)).to(device)

    global_operator = GlobalOperator(
        bundle_size=6, window_size=1,
        u_dim=2, width=16, depth=4,
        x_modes=12, y_modes=12,
    ).to(device)

    global_context, global_output = global_operator(input=global_input)
    print(f'global_context: {global_context.shape}')
    print(f'global_output: {global_output.shape}')

    local_operator = LocalOperator(
        bundle_size=6, window_size=1,
        u_dim=2, 
        width=global_operator.width, depth=2,
        x_modes=33, y_modes=33,
        x_res=64, y_res=64,
    ).to(device)
    
    local_output: torch.Tensor = local_operator(
        input=local_input, global_context=global_context
    )
    print(f'local_output: {local_output.shape}')




