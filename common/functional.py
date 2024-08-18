
import torch


def compute_velocity_field(input: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Compute the velocity field along the a dimension of any input tensor

    Parameters:
        - tensor (torch.Tensor): Input tensor
        - dim (int): The axis along which the velocity field is computed

    Returns:
        torch.Tensor
    """
    output: torch.Tensor = (input ** 2).sum(dim=dim, keepdim=True) ** 0.5
    assert output.shape[dim] == 1
    return output


def retrieve_pooled_values(
    source: torch.Tensor, pooling_indices: torch.Tensor
) -> torch.Tensor:
    """
    Constructs a new tensor from the source tensor, where only the elements 
    at the positions specified by the pooling indices are retained. The 
    resulting tensor has the same shape as pooling_indices.

    Parameters:
        - source (torch.Tensor): 
            The source tensor from which to retrieve values.
        - pooling_indices (torch.Tensor): 
            The pooling indices obtained from a max pooling operation on another tensor.
    
    Returns:
        torch.Tensor: 
        A tensor with the same shape as pooling_indices, containing values from the 
            source tensor at the pooling positions.
    """
    assert source.ndim == pooling_indices.ndim == 4
    assert pooling_indices.shape[-2:] == (source.shape[-2] // 2, source.shape[-1] // 2)
    source_flat = source.view(-1)
    indices_flat = pooling_indices.view(-1)
    # Retrieve values from the source tensor at the pooling indices
    pooled_values_flat = source_flat[indices_flat]
    return pooled_values_flat.view_as(pooling_indices).to(dtype=source.dtype)

