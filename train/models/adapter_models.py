'''
    Adapter model
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LinearAdapterWithLayerNorm(nn.Module):
    """
    Linear adapter with layer normalization for feature transformation.
    Used to connect IDS text embedder output to ControlNet input.
    """
    
    def __init__(self, hidden_dim: int, projection_dim: int):
        """
        Initialize the linear adapter.
        
        Args:
            hidden_dim (int): Input feature dimension (from IDS text embedder)
            projection_dim (int): Output feature dimension (for ControlNet)
        """
        super(LinearAdapterWithLayerNorm, self).__init__()
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        
        self.linear = nn.Linear(hidden_dim, projection_dim)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the adapter.
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., hidden_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (..., projection_dim)
        """
        # Input first passes through the linear layer
        x = self.linear(x)
        # Then through layer normalization
        x = self.layer_norm(x)
        return x
class MLPAdapter(nn.Module):
    """
    Multi-layer perceptron adapter for more complex feature transformation.
    Alternative to LinearAdapterWithLayerNorm for better feature mapping.
    """
    
    def __init__(self, hidden_dim: int, projection_dim: int, 
                 intermediate_dim: Optional[int] = None, dropout: float = 0.1):
        """
        Initialize the MLP adapter.
        
        Args:
            hidden_dim (int): Input feature dimension
            projection_dim (int): Output feature dimension
            intermediate_dim (int, optional): Intermediate dimension. Defaults to hidden_dim * 2
            dropout (float): Dropout rate
        """
        super(MLPAdapter, self).__init__()
        
        if intermediate_dim is None:
            intermediate_dim = hidden_dim * 2
            
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP adapter.
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., hidden_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (..., projection_dim)
        """
        return self.mlp(x)

