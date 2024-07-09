import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Reference:

[1] Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2020). 
    Fourier neural operator for parametric partial differential equations. arXiv preprint arXiv:2010.08895.

"""


class SpectralConv2d(nn.Module):
    def __init__(
        self, 
        u_dim: int,
        x_modes: int, 
        y_modes: int,
    ):
        super().__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.

        Parameters
        ---
        u_dim (int): number of physical fields
        x_modes (int): number of Fourier waves in first spatial dimension (x)
        y_modes (int): number of Fourier waves in second spatial dimension (y)
        """

        self.u_dim: int = u_dim

        # expected input spatial dimension: (M, N)
        self.x_modes: int = x_modes   # must be less than M
        self.y_modes: int = y_modes   # must be less than floor(N/2) + 1 due to the use of torch.rfft2

        self.scale: float = 1 / (u_dim * u_dim)
        self.weights_R = nn.Parameter(
            data=self.scale * torch.rand(u_dim, u_dim, x_modes, y_modes, dtype=torch.cfloat)
        )

    def R(self, input: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", input, self.weights_R)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 4, 'Expected input of shape: (batch_size, self.u_dim, x_dim, y_dim)'
        assert input.shape[1] == self.u_dim, 'input.shape[1] must match self.u_dim'

        batch_size: int = input.shape[0]
        x_dim: int = input.shape[2]
        y_dim: int = input.shape[3]
        out_y_dim: int = y_dim // 2 + 1

        # Fourier coeffcients
        out_fft: torch.Tensor = torch.fft.rfft2(input)
        assert out_fft.shape == (batch_size, self.u_dim, x_dim, out_y_dim)

        # Linear transformation
        out_linear: torch.Tensor= self.R(
            input=out_fft[:, :, :self.x_modes, :self.y_modes],
        )
        # Inverse Fourier transform
        out_ifft: torch.Tensor = torch.fft.irfft2(out_linear, s=(x_dim, y_dim))
        return out_ifft


class FourierLayer(nn.Module):

    def __init__(self, spectral_conv2d: SpectralConv2d, W_width) -> None:
        self.local_linear = nn.Conv2d(in_channels=)


class FNO2d(nn.Module):
    def __init__(
        self, 
        u_dim: int, 
        x_modes: int = 12, 
        y_modes: int = 12, 
        width_W: int = 20, 
        initial_steps: int = 10,
    ):
        super().__init__()

        """
        Parameters:
        ---

        u_dim (int): dim of the physic field
        x_modes (int): number of Fourier waves in the first dimension
        y_modes (int): number of Fourier waves in the second dimension
        W_width (int): 




        """

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batch_size, x, y, c)
        output: the solution of the next timestep
        output shape: (batch_size, x, y, c)
        """

        self.x_modes: int = x_modes
        self.y_modes: int = y_modes
        self.width: int = width
        self.padding: int = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(in_features=initial_steps * u_dim + 2, out_features=self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2d(in_channels=self.width, out_channels=self.width, x_modes=self.x_modes, y_modes=self.y_modes)
        self.conv1 = SpectralConv2d(in_channels=self.width, out_channels=self.width, x_modes=self.x_modes, y_modes=self.y_modes)
        self.conv2 = SpectralConv2d(in_channels=self.width, out_channels=self.width, x_modes=self.x_modes, y_modes=self.y_modes)
        self.conv3 = SpectralConv2d(in_channels=self.width, out_channels=self.width, x_modes=self.x_modes, y_modes=self.y_modes)
        self.w0 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=1)
        self.w1 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=1)
        self.w2 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=1)
        self.w3 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=1)

        self.fc1 = nn.Linear(in_features=self.width, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=u_dim)

    def forward(self, x: torch.Tensor, grid: torch.Tensor):
        # x dim = [b, x1, x2, t*v]
        batch_size: int = x.shape[0]

        
        x = torch.cat(tensors=(x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        
        # Pad tensor with boundary condition
        x = F.pad(input=x, pad=[0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding] # Unpad the tensor
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        return x.unsqueeze(-2)
    

if __name__ == '__main__':
    x = torch.rand(32, 2, 64, 64)
    self = SpectralConv2d(u_dim=x.shape[1], x_modes=16, y_modes=16)
    y = self(x)


