import torch
from torch.nn.functional import mse_loss as torch_mse_loss

def mse_loss(output: torch.Tensor,
             target: torch.Tensor) -> torch.Tensor:
    return torch_mse_loss(output, target)