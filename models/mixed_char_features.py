"""
Learnable Feature Mixer for combining OCR and IDS features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableFeatureMixer(nn.Module):
    """
    Learnable feature mixer that combines OCR and IDS features
    mixed = sigmoid(alpha_logit) * ocr + (1 - sigmoid(alpha_logit)) * ids
    """
    
    def __init__(self, alpha_init: float = 0.0):
        """
        Args:
            alpha_init: Initial value for alpha_logit (default 0.0 -> sigmoid(0.0) = 0.5)
        """
        super().__init__()
        # Use Parameter to make it learnable
        self.alpha_logit = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
    
    def forward(self, ocr_features: torch.Tensor, ids_features: torch.Tensor) -> torch.Tensor:
        """
        Mix OCR and IDS features with learnable alpha
        
        Args:
            ocr_features: OCR feature vectors, shape (..., feature_dim)
            ids_features: IDS feature vectors, shape (..., feature_dim)
            
        Returns:
            mixed_features: Mixed feature vectors, same shape as inputs
        """
        # Ensure alpha is in [0, 1] range using sigmoid
        alpha = torch.sigmoid(self.alpha_logit)
        
        # Mix features
        mixed = alpha * ocr_features + (1 - alpha) * ids_features
        
        return mixed
    
    def get_alpha(self) -> float:
        """Get current alpha value (for monitoring)"""
        return torch.sigmoid(self.alpha_logit).item()

