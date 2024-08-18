from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.functional import retrieve_pooled_values


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

        self.scale = 1 / (u_dim * u_dim)
        self.weights1 = nn.Parameter(self.scale * torch.rand(u_dim, u_dim, self.x_modes, self.y_modes, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(u_dim, u_dim, self.x_modes, self.y_modes, dtype=torch.cfloat))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # "Discrete Case and FFT" (page 5) in the paper
        output = torch.zeros(
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
        weights = torch.rand(in_timesteps, out_timesteps, dtype=torch.float)
        self.weights = nn.Parameter(weights)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 5
        assert input.shape[1] == self.in_timesteps
        batch_size, in_timesteps, u_dim, x_res, y_res = input.shape
        output: torch.Tensor = torch.einsum("biuxy,io->bouxy", input, self.weights)
        weight_sum: torch.Tensor = self.weights.sum(dim=0).reshape(1, self.out_timesteps, 1, 1, 1)
        output: torch.Tensor = output / weight_sum  # broadcasted to input shape
        assert output.shape == (batch_size, self.out_timesteps, u_dim, x_res, y_res)
        return output


class ComplexMaxPool2d(nn.Module):
    """
    Not used
    """
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert input.ndim == 5
        # assert input.shape[3] % 2 == input.shape[4] % 2 == 0
        batch_size, t_dim, u_dim, x_res, y_res = input.shape
        power_spectrum: torch.Tensor = input.abs()
        power_spectrum = power_spectrum.reshape(batch_size, t_dim * u_dim, x_res, y_res)
        pooling_indices: torch.Tensor
        _, pooling_indices = self.pool(power_spectrum)
        output: torch.Tensor = retrieve_pooled_values(
            source=input.reshape(batch_size, t_dim * u_dim, x_res, y_res), 
            pooling_indices=pooling_indices,
        )
        assert output.shape == pooling_indices.shape
        return (
            output.reshape(batch_size, t_dim, u_dim, x_res // 2, y_res // 2), 
            pooling_indices.reshape(batch_size, t_dim, u_dim, x_res // 2, y_res // 2)
        )


class ComplexMaxUnpool2d(nn.Module):
    """
    Not used
    """
    def __init__(self):
        super().__init__()
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, input: torch.Tensor, pooling_indices: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 5
        batch_size, t_dim, u_dim, x_res, y_res = input.shape
        input: torch.Tensor = input.reshape(batch_size, t_dim * u_dim, x_res, y_res)
        pooling_indices: torch.Tensor = pooling_indices.reshape(batch_size, t_dim * u_dim, x_res, y_res)
        real_part: torch.Tensor = self.unpool(
            input=input.real, indices=pooling_indices, 
            output_size=(batch_size, t_dim * u_dim, x_res * 2, y_res * 2),
        )
        imag_part: torch.Tensor = self.unpool(
            input=input.imag, indices=pooling_indices, 
            output_size=(batch_size, t_dim * u_dim, x_res * 2, y_res * 2),
        )
        output: torch.Tensor = torch.complex(real=real_part, imag=imag_part)
        return output.reshape(batch_size, t_dim, u_dim, x_res * 2, y_res * 2)


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
        fourier_coeff: torch.Tensor = self.R(input=fourier_coeff)
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
        self.scale = 1 / (u_dim * u_dim)
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
        original_shape = input.shape
        # Move normalized dimensions to the end
        permute_order: List[int] = [d for d in range(len(original_shape)) if d not in self.dims] + list(self.dims)
        input: torch.Tensor = input.permute(*permute_order)
        # Apply LayerNorm on the last dimensions
        output: torch.Tensor = self.layer_norm(input)
        # Permute back to the original shape
        inverse_permute_order: List[int] = [permute_order.index(i) for i in range(len(original_shape))]
        output: torch.Tensor = output.permute(*inverse_permute_order)
        assert output.shape == original_shape
        return output


class ContextAggregateLayer(nn.Module):

    def __init__(
        self,
        out_x_res: int,
        out_y_res: int,
    ):
        super().__init__()
        self.out_x_res: int = out_x_res
        self.out_y_res: int = out_y_res

        self.weights = nn.Parameter(data=torch.randn(self.out_x_res, self.out_y_res))

    def forward(self, global_context: torch.Tensor, local_context: torch.Tensor) -> torch.Tensor:
        # (batch_size, timesteps, context_dim, x_res, y_res)
        assert global_context.ndim == local_context.ndim == 5
        assert global_context.shape[:3] == local_context.shape[:3]
        batch_size, timesteps, context_dim = global_context.shape[:3]

        # Interpolation to local resolution
        global_context: torch.Tensor = ContextAggregateLayer._transform_resolution(
            input=global_context,
            target_x_res=local_context.shape[3], target_y_res=local_context.shape[4]
        )
        assert global_context.shape == local_context.shape == (
            batch_size, timesteps, context_dim, self.out_x_res, self.out_y_res,
        )
        # TODO: Improve
        output: torch.Tensor = (
            global_context * torch.sigmoid(self.weights) + local_context    # broadcasted
        )
        return output

    @staticmethod
    def _transform_resolution(input: torch.Tensor, target_x_res: int, target_y_res: int) -> torch.Tensor:
        output: torch.Tensor = input.reshape(
            input.shape[0], input.shape[1] * input.shape[2], input.shape[3], input.shape[4],
        )
        output: torch.Tensor = F.interpolate(
            input=output,
            size=(target_x_res, target_y_res),
            mode='bicubic',
        )
        output: torch.Tensor = output.reshape(
            input.shape[0], input.shape[1], input.shape[2], target_x_res, target_y_res,
        )
        return output
    

