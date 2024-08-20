from typing import List, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyLinearTransformation(nn.Module):

    """
    Extended version of Layer R in paper: https://arxiv.org/pdf/2010.08895
    NOTE: u_dim is d_v in the paper
    """

    def __init__(self, u_dim: int, x_modes: int, y_modes: int):
        super().__init__()
        self.u_dim: int = u_dim
        self.x_modes: int = x_modes
        self.y_modes: int = y_modes

        self.scale: float = 1 / (u_dim * u_dim)
        self.weights1 = nn.Parameter(self.scale * torch.rand(u_dim, u_dim, self.x_modes, self.y_modes, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(u_dim, u_dim, self.x_modes, self.y_modes, dtype=torch.cfloat))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # "Discrete Case and FFT" (page 5) in the paper
        output: torch.Tensor = torch.zeros(
            input.shape[0], input.shape[1], self.u_dim, input.shape[-2], input.shape[-1], 
            dtype=torch.cfloat, device=input.device,
        )
        output[:, :, :, :self.x_modes, :self.y_modes] = self.linear_tranform(
            input[:, :, :, :self.x_modes, :self.y_modes], 
            self.weights1
        )
        output[:, :, :, -self.x_modes:, :self.y_modes] = self.linear_tranform(
            input[:, :, :, -self.x_modes:, :self.y_modes], 
            self.weights2
        )
        return output

    @staticmethod
    def linear_tranform(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
        return torch.einsum('btixy,ioxy->btoxy', tensor1, tensor2)


class TemporalAggregateLayer(nn.Module):

    def __init__(self, in_timesteps: int, out_timesteps: int):
        super().__init__()
        self.in_timesteps: int = in_timesteps
        self.out_timesteps: int = out_timesteps
        weights: torch.Tensor = torch.rand(in_timesteps, out_timesteps, dtype=torch.float)
        self.weights = nn.Parameter(weights)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 5
        assert input.shape[1] == self.in_timesteps
        batch_size, in_timesteps, u_dim, x_res, y_res = input.shape
        output: torch.Tensor = torch.einsum("biuxy,io->bouxy", input, self.weights)
        weight_sum: torch.Tensor = self.weights.sum(dim=0).reshape(1, self.out_timesteps, 1, 1, 1)
        output = output / weight_sum  # broadcasted to input shape
        assert output.shape == (batch_size, self.out_timesteps, u_dim, x_res, y_res)
        return output


class TemporalAttention(nn.Module):

    def __init__(
        self, 
        in_timesteps: int, out_timesteps: int, 
        width: int, 
        n_heads: int, dropout: float,
    ):
        super().__init__()
        self.in_timesteps: int = in_timesteps
        self.out_timesteps: int = out_timesteps
        self.width: int = width
        self.n_heads: int = n_heads
        self.dropout: float = dropout

        self.attention = nn.MultiheadAttention(
            embed_dim=self.width,
            num_heads=self.n_heads,
            dropout=self.dropout,
            bias=True,
        )
        self.query_projection = nn.Linear(
            in_features=self.in_timesteps, out_features=self.out_timesteps,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size: int = input.shape[0]
        x_res, y_res = input.shape[-2:]
        assert input.ndim == 5
        output: torch.Tensor = input.mean(dim=(-2, -1))
        assert output.shape == (batch_size, self.in_timesteps, self.width)

        query: torch.Tensor = self.query_projection(input=output.transpose(1, 2)).permute(2, 0, 1)
        assert query.shape == (self.out_timesteps, batch_size, self.width)

        # Seft-attention on width and resolution
        # In each batch, there are `self.out_timesteps` queries and `self.in_timesteps` key-value pairs
        # Each output timestep attends to different input timesteps
        # The attention weight for each sample in the batch is in shape (self.out_timesteps, self.in_timesteps)
        # The attention output for each sample in the batch is in shape (self.out_timesteps, self.width)
        attention_output: torch.Tensor; attention_weight: torch.Tensor
        attention_output, attention_weight = self.attention(query=query, key=output, value=output, need_weights=True)
        assert attention_output.shape == (self.out_timesteps, batch_size, self.width)
        assert attention_weight.shape == (batch_size, self.out_timesteps, self.in_timesteps)
        output = torch.einsum('biwxy,boi->bowxy', input, attention_weight)
        assert output.shape == (batch_size, self.out_timesteps, self.width, x_res, y_res)
        return output


class TemporalAttention(nn.Module):

    def __init__(
        self, 
        in_timesteps: int, out_timesteps: int, 
        width: int, x_res: int, y_res: int,
        downsampling_factor: int,
        n_heads: int, dropout: float,
    ):
        super().__init__()
        self.in_timesteps: int = in_timesteps
        self.out_timesteps: int = out_timesteps
        self.width: int = width
        self.x_res: int = x_res
        self.y_res: int = y_res
        self.n_heads: int = n_heads
        self.dropout: float = dropout

        self.downsampling_factor: int = downsampling_factor
        self.downsampled_x_res: int = self.x_res // self.downsampling_factor
        self.downsampled_y_res: int = self.y_res // self.downsampling_factor
        self.n: int = int(math.log2(self.downsampling_factor))

        self.embedding_dim: int = self.width * self.downsampled_x_res * self.downsampled_y_res

        self.attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=self.n_heads,
            dropout=self.dropout,
            bias=True,
        )
        self.query_projection = nn.Linear(
            in_features=self.in_timesteps, out_features=self.out_timesteps,
        )
        # equivalent to applying Conv2d with kernel_size=2, stride=2, padding=0 with no activations `n` times
        self.spatial_conv = nn.Conv2d(
            in_channels=self.in_timesteps * self.width,
            out_channels=self.in_timesteps * self.width,
            kernel_size=2 + (self.n - 1), # k_effective = k + (n - 1) * (k - 1)
            stride=2 ** self.n, # s_effective = s ** n
            padding=0 * self.n, # p_effective = p * n
            bias=False,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size: int = input.shape[0]
        assert input.ndim == 5
        assert input.shape == (batch_size, self.in_timesteps, self.width, self.x_res, self.y_res)

        output: torch.Tensor = input.reshape(batch_size, self.in_timesteps * self.width, self.x_res, self.y_res)
        output = self.spatial_conv(input=output)
        output = output.reshape(batch_size, self.in_timesteps, self.embedding_dim)
        output = output.transpose(1, 0) # (self.in_timesteps, batch_size, embedding_dim)

        query: torch.Tensor = self.query_projection(input=output.transpose(0, 2)).transpose(0, 2)
        assert query.shape == (self.out_timesteps, batch_size, self.embedding_dim)

        # Seft-attention on width and resolution
        # In each batch, there are `self.out_timesteps` queries and `self.in_timesteps` key-value pairs
        # Each output timestep attends to different input timesteps
        # The attention weight for each sample in the batch is in shape (self.out_timesteps, self.in_timesteps)
        # The attention output for each sample in the batch is in shape (self.out_timesteps, self.width)
        attention_output: torch.Tensor; attention_weight: torch.Tensor
        attention_output, attention_weight = self.attention(query=query, key=output, value=output, need_weights=True)
        assert attention_output.shape == (self.out_timesteps, batch_size, self.embedding_dim)
        assert attention_weight.shape == (batch_size, self.out_timesteps, self.in_timesteps)
        output = torch.einsum('biwxy,boi->bowxy', input, attention_weight)
        assert output.shape == (batch_size, self.out_timesteps, self.width, self.x_res, self.y_res)
        return output


class AdaptiveSpectralConv2d(nn.Module):

    def __init__(
        self, u_dim: int, x_modes: int, y_modes: int):
        """
        Adaptive 2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """
        super().__init__()
        self.u_dim: int = u_dim
        self.x_modes: int = x_modes
        self.y_modes: int = y_modes

        self.R = FrequencyLinearTransformation(u_dim=u_dim, x_modes=x_modes, y_modes=y_modes)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 5
        assert input.shape[2] == self.u_dim

        # Fourier coeffcients
        fourier_coeff: torch.Tensor = torch.fft.rfftn(input=input, dim=(3, 4))
        # Linear transformation
        fourier_coeff = self.R(input=fourier_coeff)
        # Inverse Fourier transform
        output: torch.Tensor = torch.fft.irfftn(
            input=fourier_coeff, 
            dim=(3, 4), 
            s=(input.shape[-2], input.shape[-1]), 
        )
        assert output.shape == input.shape
        return output


class LocalLinearTransformation(nn.Module):

    """
    Extended version of Layer W in paper: https://arxiv.org/pdf/2010.08895
    """

    def __init__(self, u_dim: int):
        super().__init__()
        self.u_dim: int = u_dim
        self.scale: float = 1 / (u_dim * u_dim)
        weights: torch.Tensor = self.scale * torch.rand(self.u_dim, self.u_dim)
        self.weights = nn.Parameter(weights)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 5
        assert input.shape == (input.shape[0], input.shape[1], self.u_dim, input.shape[3], input.shape[4])
        return torch.einsum('btixy,io->btoxy', input, self.weights)


class LiftingLayer(nn.Module):

    """
    Extended version of Layer P and Q in paper: https://arxiv.org/pdf/2010.08895
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.weights = nn.Parameter(
            data=torch.rand(in_features, out_features) / (in_features * out_features)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 5
        batch_size, t_dim, u_dim, x_res, y_res = input.shape
        assert self.in_features == u_dim
        output: torch.Tensor = torch.einsum('btixy,io->btoxy', input, self.weights)
        assert output.shape == (batch_size, t_dim, self.out_features, x_res, y_res)
        return output


class FeatureNormalization(nn.Module):

    def __init__(self, normalized_shape: Tuple[int, ...], dims: Tuple[int, ...]):
        super().__init__()
        assert len(normalized_shape) == len(dims)

        self.dims: Tuple[int, ...] = dims
        self.normalized_shape: Tuple[int, ...] = normalized_shape
        self.layer_norm = nn.LayerNorm(normalized_shape=normalized_shape)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim >= len(self.dims)
        original_shape: Tuple[int, ...] = input.shape
        # Move normalized dimensions to the end
        permute_order: List[int] = [d for d in range(len(original_shape)) if d not in self.dims] + list(self.dims)
        output: torch.Tensor = input.permute(*permute_order)
        # Apply LayerNorm on the last dimensions
        output = self.layer_norm(output)
        # Permute back to the original shape
        inverse_permute_order: List[int] = [permute_order.index(i) for i in range(len(original_shape))]
        output = output.permute(*inverse_permute_order)
        assert output.shape == original_shape
        return output


class ContextAggregateLayer(nn.Module):

    def __init__(
        self,
        local_x_res: int,
        local_y_res: int,
    ):
        super().__init__()
        self.local_x_res: int = local_x_res
        self.local_y_res: int = local_y_res

        self.weights = nn.Parameter(data=torch.randn(self.local_x_res, self.local_y_res))

    def forward(self, global_context: torch.Tensor, local_context: torch.Tensor) -> torch.Tensor:
        # (batch_size, timesteps, context_dim, x_res, y_res)
        assert global_context.ndim == local_context.ndim == 5
        assert global_context.shape[:3] == local_context.shape[:3]
        assert local_context.shape[-2:] == (self.local_x_res, self.local_x_res)
        batch_size, timesteps, context_dim = global_context.shape[:3]

        # Adapt to local_context resolution
        global_context = self._transform_resolution(input=global_context)
        assert global_context.shape == local_context.shape == (
            batch_size, timesteps, context_dim, self.local_x_res, self.local_y_res,
        )
        output: torch.Tensor = (
            global_context * torch.sigmoid(self.weights) + local_context    # broadcasted
        )
        return output

    def _transform_resolution(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 5
        assert input.shape[-2:] == (self.local_x_res, self.local_y_res)
        output: torch.Tensor = input.reshape(
            input.shape[0], input.shape[1] * input.shape[2], self.local_x_res, self.local_y_res,
        )
        output = F.adaptive_avg_pool2d(input=output, output_size=(self.local_x_res, self.local_y_res))
        output = output.reshape(
            input.shape[0], input.shape[1], input.shape[2], self.local_x_res, self.local_y_res,
        )
        return output
    

if __name__ == '__main__':

    self = TemporalAttention(
        in_timesteps=12, out_timesteps=6, 
        width=64, x_res=256, y_res=256,
        downsampling_factor=16,
        n_heads=16, dropout=0.1
    ).cuda()
    batch_size: int = 32
    x = torch.rand(batch_size, 12, 64, 256, 256).cuda()
    y = self(x)
    print(y.shape)




